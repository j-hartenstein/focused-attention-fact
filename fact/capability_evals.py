"""Capability evaluations (IFEval, MMLU) via lm-evaluation-harness.

Wraps lm_eval.simple_evaluate() with:
- Static val sets: pre-downloaded JSONL files committed to the repo
  (data/capability_evals/ifeval_val.jsonl, mmlu_val.jsonl)
- True random sampling for test rollouts: shuffles full HF dataset with
  different seeds, writes temp JSONL slices, runs lm_eval on each

Usage:
    # Val set (fast, deterministic — used in behavioral_validation):
    from fact.capability_evals import run_capability_eval_val
    results = run_capability_eval_val(
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        tasks=["ifeval", "mmlu"],
        val_data_dir="data/capability_evals",
        adapter_path="adapters/fact_lora",  # optional
    )

    # Test set (multiple rollouts with independent random samples):
    from fact.capability_evals import run_capability_eval_with_rollouts
    results = run_capability_eval_with_rollouts(
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        tasks=["ifeval", "mmlu"],
        n_rollouts=4,
        samples_per_rollout=250,
    )
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Any

from fact.utils import get_logger

logger = get_logger("capability_evals")

# ---------------------------------------------------------------------------
# Eval registry — add new evals here
# ---------------------------------------------------------------------------
# Each entry maps a short eval name to:
#   task        : lm_eval task name (for use without custom YAML)
#   hf_dataset  : (repo_id, config, split) for downloading full dataset
#   val_file    : filename of the static val JSONL in data/capability_evals/
#   custom_yaml : YAML template filename in data/capability_evals/
#   primary_metric  : key in lm_eval results dict to use as headline metric
#   display_metrics : {metric_key: display_name} for the comparison table
#   higher_is_better: bool
EVAL_REGISTRY: dict[str, dict[str, Any]] = {
    "ifeval": {
        "task": "ifeval",               # lm_eval task name (used to locate installed task dir)
        "hf_dataset": ("google/IFEval", None, "train"),  # for rollout downloads
        "val_file": "ifeval_val.jsonl", # static val set filename in val_data_dir
        "primary_metric": "prompt_level_strict_acc,none",
        "display_metrics": {
            "prompt_level_strict_acc,none": "IFEval prompt-strict",
            "inst_level_strict_acc,none": "IFEval inst-strict",
        },
        "higher_is_better": True,
    },
    "mmlu": {
        "task": "mmlu",
        "hf_dataset": ("cais/mmlu", "all", "validation"),
        "val_file": "mmlu_val.jsonl",
        "primary_metric": "acc,none",
        "display_metrics": {
            "acc,none": "MMLU accuracy",
        },
        "higher_is_better": True,
    },
}

# Default path to the data/capability_evals/ directory (relative to repo root)
_DEFAULT_VAL_DIR = Path(__file__).parent.parent / "data" / "capability_evals"


def _build_model_args(model_name: str, adapter_path: str | None, batch_size: int | str = 16) -> str:
    args = f"pretrained={model_name},dtype=bfloat16"
    if adapter_path:
        args += f",peft={adapter_path}"
    return args


def _get_lm_eval_task_dir(task_name: str) -> Path:
    """Return a suitable directory for writing a custom task YAML for the given task.

    For tasks with a utils.py (e.g. ifeval), we must co-locate the YAML with utils.py
    so lm_eval can resolve '!function utils.process_results'.  We find this by walking
    the installed task YAMLs for one that declares 'task: <name>'.

    For tasks with no utils.py dependency (e.g. mmlu), any task subdirectory works;
    we fall back to a known stable directory (ifeval's, since it always exists).
    """
    import lm_eval.tasks as _tasks
    import yaml as _yaml

    tasks_root = Path(_tasks.__file__).parent

    # First: look for a YAML that explicitly declares this task name
    for yaml_path in sorted(tasks_root.rglob("*.yaml")):
        try:
            content = _yaml.safe_load(yaml_path.read_text())
            if isinstance(content, dict) and content.get("task") == task_name:
                return yaml_path.parent
        except Exception:
            continue

    # Fallback: any existing task subdir (MMLU has no utils.py so location doesn't matter)
    fallback = tasks_root / task_name
    if fallback.is_dir():
        return fallback
    # Last resort: use the tasks root itself
    logger.warning(f"Could not find task dir for '{task_name}', using tasks root: {tasks_root}")
    return tasks_root


def _write_task_yaml(data_file: Path, task_dir: Path, task_suffix: str) -> tuple[Path, str]:
    """Write a self-contained task YAML alongside the installed task (so utils.py resolves).

    Returns (yaml_path, task_name_for_lm_eval).
    """
    task_name = f"fact_{task_suffix}_custom"
    yaml_path = task_dir / f"{task_name}.yaml"

    if task_suffix == "ifeval":
        content = f"""task: {task_name}
dataset_path: json
dataset_name: null
dataset_kwargs:
  data_files: "{data_file}"
output_type: generate_until
test_split: train
num_fewshot: 0
doc_to_text: prompt
doc_to_target: 0
generation_kwargs:
  until: []
  do_sample: false
  temperature: 0.0
  max_gen_toks: 512
process_results: !function utils.process_results
metric_list:
  - metric: prompt_level_strict_acc
    aggregation: mean
    higher_is_better: true
  - metric: inst_level_strict_acc
    aggregation: !function utils.agg_inst_level_acc
    higher_is_better: true
  - metric: prompt_level_loose_acc
    aggregation: mean
    higher_is_better: true
  - metric: inst_level_loose_acc
    aggregation: !function utils.agg_inst_level_acc
    higher_is_better: true
metadata:
  version: 4.0
"""
    elif task_suffix == "mmlu":
        content = f"""task: {task_name}
dataset_path: json
dataset_name: null
dataset_kwargs:
  data_files: "{data_file}"
test_split: train
output_type: multiple_choice
doc_to_text: "{{{{question.strip()}}}}\\\\nA. {{{{choices[0]}}}}\\\\nB. {{{{choices[1]}}}}\\\\nC. {{{{choices[2]}}}}\\\\nD. {{{{choices[3]}}}}\\\\nAnswer:"
doc_to_choice: ["A", "B", "C", "D"]
doc_to_target: answer
metric_list:
  - metric: acc
    aggregation: mean
    higher_is_better: true
metadata:
  version: 1.0
"""
    else:
        raise ValueError(f"No YAML template for task suffix '{task_suffix}'")

    yaml_path.write_text(content)
    return yaml_path, task_name


def _build_hflm(model_name: str, adapter_path: str | None, device: str, batch_size: int | str, model=None, tokenizer=None):
    """Build a reusable HFLM wrapper.

    If model/tokenizer are provided, wraps the already-loaded model to avoid loading
    a second copy (which would OOM on A100-40GB). Otherwise loads from model_name.

    Returns an lm_eval HFLM instance that can be passed directly to simple_evaluate().
    """
    from lm_eval.models.huggingface import HFLM
    if model is not None and tokenizer is not None:
        # Wrap existing model — no second load, no extra GPU memory.
        return HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
        )
    return HFLM(
        pretrained=model_name,
        dtype="bfloat16",
        peft=adapter_path,
        device=device,
        batch_size=batch_size,
    )


def _run_lm_eval(
    model_name: str,
    task_yaml_path: Path,
    task_name: str,
    adapter_path: str | None,
    device: str,
    batch_size: int | str,
    seed: int,
    lm=None,  # pre-built HFLM instance; if provided, model_name/adapter_path are ignored
) -> dict[str, Any]:
    """Run lm_eval.simple_evaluate() loading a single task from its YAML path."""
    import lm_eval
    from lm_eval.tasks import TaskManager

    # include_path tells TaskManager to also scan this directory for task YAMLs
    task_manager = TaskManager(include_path=str(task_yaml_path.parent))

    if lm is not None:
        results = lm_eval.simple_evaluate(
            model=lm,
            tasks=[task_name],
            task_manager=task_manager,
            batch_size=batch_size,
            random_seed=seed,
            numpy_random_seed=seed,
            torch_random_seed=seed,
            log_samples=False,
        )
    else:
        results = lm_eval.simple_evaluate(
            model="hf",
            model_args=_build_model_args(model_name, adapter_path, batch_size),
            tasks=[task_name],
            task_manager=task_manager,
            device=device,
            batch_size=batch_size,
            random_seed=seed,
            numpy_random_seed=seed,
            torch_random_seed=seed,
            log_samples=False,
        )
    return results


def _extract_metrics(
    raw: dict[str, Any],
    eval_name: str,
    task_name: str,
) -> dict[str, float]:
    """Pull display_metrics from lm_eval results dict."""
    reg = EVAL_REGISTRY[eval_name]
    task_results = raw["results"].get(task_name, {})
    out: dict[str, float] = {}
    for metric_key in reg["display_metrics"]:
        val = task_results.get(metric_key)
        if val is not None:
            out[metric_key] = float(val)
        else:
            logger.warning(
                f"Metric '{metric_key}' not found for '{eval_name}'. "
                f"Available: {list(task_results.keys())}"
            )
    return out


# ---------------------------------------------------------------------------
# Val set evaluation (uses static pre-downloaded JSONL files)
# ---------------------------------------------------------------------------

def run_capability_eval_val(
    model_name: str,
    tasks: list[str],
    val_data_dir: str | Path | None = None,
    adapter_path: str | None = None,
    device: str = "cuda",
    batch_size: int | str = "auto",
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Evaluate on the static val set (deterministic, used in behavioral_validation).

    Args:
        model_name: HuggingFace model ID.
        tasks: List of eval names from EVAL_REGISTRY (e.g. ["ifeval", "mmlu"]).
        val_data_dir: Directory containing val JSONL files and custom YAML templates.
                      Defaults to data/capability_evals/ in the repo root.
        adapter_path: Optional path to PEFT adapter directory.
        device: Device string.
        batch_size: Batch size ("auto" for automatic sizing).
        seed: Random seed (does not affect which samples are evaluated — that's
              fixed by the static val set files).

    Returns:
        {eval_name: {metric_key: value}}
    """
    val_dir = Path(val_data_dir) if val_data_dir else _DEFAULT_VAL_DIR

    for t in tasks:
        if t not in EVAL_REGISTRY:
            raise ValueError(f"Unknown eval '{t}'. Available: {list(EVAL_REGISTRY.keys())}")
        val_file = val_dir / EVAL_REGISTRY[t]["val_file"]
        if not val_file.exists():
            raise FileNotFoundError(
                f"Val set not found: {val_file}\n"
                f"Run: python data/scripts/download_capability_evals.py"
            )

    out: dict[str, dict[str, float]] = {}
    written_yamls: list[Path] = []
    try:
        for t in tasks:
            reg = EVAL_REGISTRY[t]
            val_file = val_dir / reg["val_file"]

            logger.info(f"  {t} val: {val_file} ({sum(1 for _ in open(val_file))} samples)")
            task_dir = _get_lm_eval_task_dir(reg["task"])
            task_yaml, task_name = _write_task_yaml(val_file, task_dir, t)
            written_yamls.append(task_yaml)

            raw = _run_lm_eval(
                model_name=model_name,
                task_yaml_path=task_yaml,
                task_name=task_name,
                adapter_path=adapter_path,
                device=device,
                batch_size=batch_size,
                seed=seed,
            )
            out[t] = _extract_metrics(raw, t, task_name)
            logger.info(f"  {t}: {out[t]}")
    finally:
        for p in written_yamls:
            p.unlink(missing_ok=True)

    return out


# ---------------------------------------------------------------------------
# Fast native IFEval scorer (bypasses lm_eval generate_until)
# ---------------------------------------------------------------------------

def _run_ifeval_native(
    model,
    tokenizer,
    items: list[dict],
    max_new_tokens: int = 256,
    batch_size: int = 16,
) -> dict[str, float]:
    """Score IFEval using our fast batched generator + lm_eval's scoring functions.

    Generates responses with generate_responses_batched (properly batched on GPU),
    then scores with test_instruction_following_strict directly — no lm_eval pipeline.

    Returns {metric_key: value} matching EVAL_REGISTRY["ifeval"]["display_metrics"].
    """
    from lm_eval.tasks.ifeval.utils import InputExample, test_instruction_following_strict
    from fact.evaluate import generate_responses_batched

    prompts = [item["prompt"] for item in items]
    responses = generate_responses_batched(
        model, tokenizer, prompts,
        max_new_tokens=max_new_tokens, batch_size=batch_size,
    )

    prompt_strict_results = []
    inst_strict_results = []
    for item, response in zip(items, responses):
        inp = InputExample(
            key=item["key"],
            instruction_id_list=item["instruction_id_list"],
            prompt=item["prompt"],
            kwargs=item["kwargs"],
        )
        out = test_instruction_following_strict(inp, response)
        prompt_strict_results.append(float(out.follow_all_instructions))
        inst_strict_results.extend([float(v) for v in out.follow_instruction_list])

    prompt_acc = sum(prompt_strict_results) / len(prompt_strict_results)
    inst_acc = sum(inst_strict_results) / len(inst_strict_results) if inst_strict_results else 0.0
    return {
        "prompt_level_strict_acc,none": prompt_acc,
        "inst_level_strict_acc,none": inst_acc,
    }


# ---------------------------------------------------------------------------
# Test set evaluation with true random rollouts
# ---------------------------------------------------------------------------

def _load_full_dataset(eval_name: str) -> list[dict]:
    """Download full HF dataset once and return as a list. Cached by HF datasets library."""
    from datasets import load_dataset

    reg = EVAL_REGISTRY[eval_name]
    repo_id, config, split = reg["hf_dataset"]
    logger.info(f"  Loading {repo_id} ({split}) ...")

    load_kwargs: dict[str, Any] = {"split": split}
    if config:
        load_kwargs["name"] = config

    ds = load_dataset(repo_id, **load_kwargs)
    return list(ds)


def _normalise_row(eval_name: str, item: dict) -> dict:
    """Normalise a dataset row to the JSONL schema used by the custom task YAML."""
    if eval_name == "ifeval":
        return {
            "key": item["key"],
            "prompt": item["prompt"],
            "instruction_id_list": item["instruction_id_list"],
            "kwargs": item["kwargs"],
        }
    if eval_name == "mmlu":
        return {
            "question": item["question"],
            "subject": item["subject"],
            "choices": list(item["choices"]),
            "answer": int(item["answer"]),
        }
    return dict(item)


def _write_rollout_jsonl(
    eval_name: str,
    items: list[dict],
    n: int,
    seed: int,
    tmp_dir: str,
) -> Path:
    """Shuffle items with seed, take n, write to a temp JSONL. No network I/O."""
    import random

    rng = random.Random(seed)
    shuffled = items[:]
    rng.shuffle(shuffled)
    subset = shuffled[:n]

    out_path = Path(tmp_dir) / f"{eval_name}_rollout_seed{seed}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for item in subset:
            f.write(json.dumps(_normalise_row(eval_name, item), ensure_ascii=False) + "\n")
    return out_path


def run_capability_eval_with_rollouts(
    model_name: str,
    tasks: list[str],
    n_rollouts: int = 4,
    samples_per_rollout: int = 250,
    adapter_path: str | None = None,
    device: str = "cuda",
    base_seed: int = 0,
    batch_size: int | str = "auto",
    val_data_dir: str | Path | None = None,
    model=None,      # pre-loaded transformers model (used for fast native IFEval generation)
    tokenizer=None,  # pre-loaded tokenizer
    ifeval_batch_size: int = 16,
    ifeval_max_new_tokens: int = 256,
) -> dict[str, dict[str, dict[str, float]]]:
    """Run multiple rollouts with truly independent random samples, compute mean + 95% CI.

    Each rollout downloads the full benchmark, shuffles with a different seed, takes
    samples_per_rollout items, and evaluates. Seeds are base_seed, base_seed+1, ...

    IFEval is scored natively (generate_responses_batched + direct scoring) when
    model and tokenizer are provided, bypassing lm_eval's slow generate_until pipeline.
    MMLU always uses lm_eval (loglikelihood, already fast).

    Args:
        model_name: HuggingFace model ID.
        tasks: List of eval names from EVAL_REGISTRY.
        n_rollouts: Number of independent evaluation runs (default: 4).
        samples_per_rollout: Samples per rollout (default: 250).
        adapter_path: Optional PEFT adapter path.
        device: Device string.
        base_seed: First seed (default: 0); seeds are base_seed + i.
        batch_size: Batch size for lm_eval (MMLU).
        val_data_dir: Directory containing custom YAML templates.
        model: Pre-loaded transformers model for fast IFEval generation (optional).
        tokenizer: Pre-loaded tokenizer (required if model is provided).
        ifeval_batch_size: Batch size for native IFEval generation (default: 16).
        ifeval_max_new_tokens: Max tokens for IFEval responses (default: 256).

    Returns:
        {eval_name: {metric_key: {"mean", "std", "ci95_low", "ci95_high", "values"}}}
    """
    val_dir = Path(val_data_dir) if val_data_dir else _DEFAULT_VAL_DIR

    for t in tasks:
        if t not in EVAL_REGISTRY:
            raise ValueError(f"Unknown eval '{t}'. Available: {list(EVAL_REGISTRY.keys())}")

    use_native_ifeval = model is not None and tokenizer is not None
    if "ifeval" in tasks:
        if use_native_ifeval:
            logger.info("  IFEval: using native batched generation (fast path)")
        else:
            logger.info("  IFEval: model/tokenizer not provided, falling back to lm_eval generate_until (slow)")

    all_results: list[dict[str, dict[str, float]]] = []

    # Download each dataset once (HF caches to disk; list() loads into RAM once per call)
    full_datasets: dict[str, list[dict]] = {}
    for t in tasks:
        full_datasets[t] = _load_full_dataset(t)
        logger.info(f"  {t}: {len(full_datasets[t])} total examples available")

    # Build HFLM only for tasks that need lm_eval (i.e. mmlu or ifeval fallback)
    lm_eval_tasks = [t for t in tasks if t != "ifeval" or not use_native_ifeval]
    lm = None
    if lm_eval_tasks:
        logger.info("  Building HFLM wrapper for lm_eval tasks...")
        lm = _build_hflm(model_name, adapter_path, device, batch_size, model=model, tokenizer=tokenizer)

    for i in range(n_rollouts):
        seed = base_seed + i
        logger.info(f"=== Rollout {i + 1}/{n_rollouts} (seed={seed}) ===")
        rollout_out: dict[str, dict[str, float]] = {}

        written_yamls: list[Path] = []
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                for t in tasks:
                    import random as _random
                    rng = _random.Random(seed)
                    shuffled = full_datasets[t][:]
                    rng.shuffle(shuffled)
                    subset = shuffled[:samples_per_rollout]
                    items = [_normalise_row(t, row) for row in subset]

                    if t == "ifeval" and use_native_ifeval:
                        rollout_out[t] = _run_ifeval_native(
                            model, tokenizer, items,
                            max_new_tokens=ifeval_max_new_tokens,
                            batch_size=ifeval_batch_size,
                        )
                        logger.info(f"  {t}: {rollout_out[t]}")
                    else:
                        # Write JSONL and run via lm_eval (used for mmlu + ifeval fallback)
                        data_file = Path(tmp_dir) / f"{t}_rollout_seed{seed}.jsonl"
                        with open(data_file, "w", encoding="utf-8") as f:
                            for row in items:
                                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        reg = EVAL_REGISTRY[t]
                        task_dir = _get_lm_eval_task_dir(reg["task"])
                        task_yaml, task_name = _write_task_yaml(data_file, task_dir, t)
                        written_yamls.append(task_yaml)

                        raw = _run_lm_eval(
                            model_name=model_name,
                            task_yaml_path=task_yaml,
                            task_name=task_name,
                            adapter_path=adapter_path,
                            device=device,
                            batch_size=batch_size,
                            seed=seed,
                            lm=lm,
                        )
                        rollout_out[t] = _extract_metrics(raw, t, task_name)
                        logger.info(f"  {t}: {rollout_out[t]}")
                # tmp_dir (JSONL files) deleted here automatically
        finally:
            for p in written_yamls:
                p.unlink(missing_ok=True)

        all_results.append(rollout_out)

    # Aggregate: mean + 95% CI
    out: dict[str, dict[str, dict[str, float]]] = {}
    for t in tasks:
        out[t] = {}
        for metric_key in EVAL_REGISTRY[t]["display_metrics"]:
            values = [r[t].get(metric_key, float("nan")) for r in all_results]
            values_clean = [v for v in values if not math.isnan(v)]
            if not values_clean:
                continue
            mean = sum(values_clean) / len(values_clean)
            if len(values_clean) > 1:
                variance = sum((v - mean) ** 2 for v in values_clean) / (len(values_clean) - 1)
                std = math.sqrt(variance)
                ci95 = 1.96 * std / math.sqrt(len(values_clean))
            else:
                std = 0.0
                ci95 = 0.0
            out[t][metric_key] = {
                "mean": mean,
                "std": std,
                "ci95_low": mean - ci95,
                "ci95_high": mean + ci95,
                "values": values_clean,
            }
    # Explicitly release HFLM before returning so GPU memory is freed
    # before the caller loads the next model.
    if lm is not None:
        del lm
    return out


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_capability_comparison(
    base_results: dict[str, dict[str, float]],
    fact_results: dict[str, dict[str, float]],
    ce_cap: dict[str, dict[str, float]] | None = None,
) -> str:
    """Format a comparison table for capability eval results.

    If ce_cap is provided, shows Base / CE / FACT with both delta columns.
    Otherwise shows the two-column Base / FACT table.
    """
    lines: list[str] = []
    if ce_cap is not None:
        lines.append(f"\n{'Metric':<30} {'Base':>10} {'CE':>10} {'FACT':>10} {'Δ CE−Base':>11} {'Δ FACT−Base':>12}")
        lines.append("-" * 85)
    else:
        lines.append(f"\n{'Metric':<30} {'Base':>10} {'FACT':>10} {'Δ (FACT−Base)':>14}")
        lines.append("-" * 66)

    for eval_name, reg in EVAL_REGISTRY.items():
        if eval_name not in base_results:
            continue
        for metric_key, display_name in reg["display_metrics"].items():
            base_val = base_results[eval_name].get(metric_key)
            fact_val = fact_results.get(eval_name, {}).get(metric_key)
            if base_val is None or fact_val is None:
                continue
            direction = "(higher better)" if reg["higher_is_better"] else "(lower better)"
            if ce_cap is not None:
                ce_val = ce_cap.get(eval_name, {}).get(metric_key)
                if ce_val is None:
                    ce_str = f"{'N/A':>10}"
                    d_ce_str = f"{'N/A':>11}"
                else:
                    ce_str = f"{ce_val:>9.2%}"
                    d_ce_str = f"{(ce_val - base_val):>+10.2%}"
                d_fact = fact_val - base_val
                lines.append(
                    f"{display_name:<30} {base_val:>9.2%} {ce_str} {fact_val:>9.2%} "
                    f"{d_ce_str} {d_fact:>+11.2%}  {direction}"
                )
            else:
                delta = fact_val - base_val
                lines.append(
                    f"{display_name:<30} {base_val:>9.2%} {fact_val:>9.2%} "
                    f"{delta:>+13.2%}  {direction}"
                )

    return "\n".join(lines)
