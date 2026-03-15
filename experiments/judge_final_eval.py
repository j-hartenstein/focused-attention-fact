"""Judge final eval responses and compute bootstrap confidence intervals.

Loads the response dump from run_final_eval.py, runs StrongREJECT judge on
jailbreak ASR responses, then bootstraps 1000x over binary outcomes (JB/syco)
and reads rollout-level scores (MMLU/IFEval) to report mean ± 95% CI for all
metrics across base / CE / FACT.

Usage:
    python experiments/judge_final_eval.py \\
        --dump results/final_eval_responses.jsonl \\
        --results results/final_eval.json \\
        --output results/final_eval_judged.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from fact.evaluate import strongreject_judge, strongreject_is_success
from fact.utils import get_logger

logger = get_logger("judge_final_eval")

N_BOOTSTRAP = 1000
CI_Z = 1.96  # 95% CI


def bootstrap_ci(binary_outcomes: list[int], n: int = N_BOOTSTRAP, seed: int = 0) -> dict:
    """Bootstrap 95% CI for a proportion from binary outcomes (0/1 list)."""
    rng = random.Random(seed)
    k = len(binary_outcomes)
    means = []
    for _ in range(n):
        sample = [binary_outcomes[rng.randrange(k)] for _ in range(k)]
        means.append(sum(sample) / k)
    means.sort()
    mean = sum(binary_outcomes) / k
    ci_low = means[int(0.025 * n)]
    ci_high = means[int(0.975 * n)]
    return {"mean": mean, "ci95_low": ci_low, "ci95_high": ci_high, "n": k}


def rollout_ci(values: list[float]) -> dict:
    """Mean ± 95% CI from a small list of rollout scores (t-based via normal approx)."""
    n = len(values)
    mean = sum(values) / n
    if n > 1:
        var = sum((v - mean) ** 2 for v in values) / (n - 1)
        std = math.sqrt(var)
        ci = CI_Z * std / math.sqrt(n)
    else:
        ci = 0.0
    return {"mean": mean, "ci95_low": mean - ci, "ci95_high": mean + ci, "n": n, "values": values}


def fmt_ci(d: dict, pct: bool = True) -> str:
    scale = 100 if pct else 1
    mean = d["mean"] * scale
    lo = d["ci95_low"] * scale
    hi = d["ci95_high"] * scale
    return f"{mean:.1f}% [{lo:.1f}, {hi:.1f}]"


def _build_plot_schema(final: dict, model_keys: list[str], asr_key: str, meta: dict) -> dict:
    """Convert judged output to the schema expected by plot_validation_results.py.

    That schema: {model_key: {jailbreak: {asr, refusal_rate, n_total}, sycophancy: {..., n_total}, capabilities: {task: {metric: value}}}}
    We use bootstrap CI means as the point estimates; n_total is set to match
    the effective sample size so Wilson CIs in the plotter are comparable.
    """
    out = {}
    n_jb = meta.get("n_jailbreak", 200)
    n_sy = meta.get("n_sycophancy", 200)
    for mk in model_keys:
        m = final[mk]
        # Pick judge-based ASR if available, else regex
        asr_val = m["jailbreak"][asr_key]["mean"]
        cap_flat: dict[str, dict[str, float]] = {}
        for eval_name, metrics in m.get("capabilities", {}).items():
            cap_flat[eval_name] = {k: v["mean"] for k, v in metrics.items()}
        out[mk] = {
            "jailbreak": {
                "asr": asr_val,
                "refusal_rate": m["jailbreak"]["refusal_rate"]["mean"],
                "n_total": n_jb,
            },
            "sycophancy": {
                "sycophancy_rate": m["sycophancy"]["sycophancy_rate"]["mean"],
                "clean_accuracy": m["sycophancy"]["clean_accuracy"]["mean"],
                "wrapped_correct_rate": m["sycophancy"]["wrapped_correct_rate"]["mean"],
                "n_total": n_sy,
            },
            "capabilities": cap_flat,
        }
    return out


def _plot_results(final: dict, model_keys: list[str], asr_key: str, out_path: str | None, meta: dict) -> None:
    """Produce the research-quality figure using plot_validation_results.py logic."""
    import subprocess
    import tempfile

    plot_data = _build_plot_schema(final, model_keys, asr_key, meta)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump(plot_data, tmp)
        tmp_path = tmp.name

    plotter = Path(__file__).parent / "plot_validation_results.py"
    cmd = [sys.executable, str(plotter), "--results", tmp_path, "--show-clean"]
    if out_path:
        cmd += ["--output", out_path]
    else:
        cmd += ["--output", "/tmp/final_eval_plot.pdf"]

    result = subprocess.run(cmd)
    Path(tmp_path).unlink(missing_ok=True)
    if result.returncode != 0:
        logger.warning("plot_validation_results.py exited with non-zero status — plot may be incomplete")


def main(args):
    # Load raw results (for capability rollout scores)
    results_path = Path(args.results)
    if not results_path.exists():
        raise FileNotFoundError(f"Results JSON not found: {results_path}")
    with open(results_path, encoding="utf-8") as f:
        raw_results = json.load(f)

    # Load response dump
    dump_path = Path(args.dump)
    if not dump_path.exists():
        raise FileNotFoundError(f"Response dump not found: {dump_path}")

    jb_rows = []
    sy_rows = []
    with open(dump_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row["task"] == "jailbreak":
                jb_rows.append(row)
            elif row["task"] == "sycophancy":
                sy_rows.append(row)

    logger.info(f"Loaded {len(jb_rows)} jailbreak rows, {len(sy_rows)} sycophancy rows")

    # Determine which models are present
    model_keys = list(raw_results.get("models", {}).keys())
    logger.info(f"Models: {model_keys}")

    # --- Run StrongREJECT judge on jailbreak ASR responses ---
    judge_verdicts: dict[str, list[dict]] = {}
    for mk in model_keys:
        resp_key = f"{mk}_asr_response"
        if not jb_rows or resp_key not in jb_rows[0]:
            logger.warning(f"No ASR responses found for model '{mk}', skipping judge")
            continue
        responses = [r[resp_key] for r in jb_rows]
        prompts = [r["wrapped_prompt"] for r in jb_rows]
        logger.info(f"Running StrongREJECT judge for {mk} ({len(responses)} responses)...")
        judge_verdicts[mk] = strongreject_judge(responses, prompts)
        logger.info(f"  Done. RPD remaining: {getattr(strongreject_judge, 'last_rpd_remaining', 'unknown')}")

    # --- Compute CIs ---
    final: dict[str, dict] = {}

    for mk in model_keys:
        model_raw = raw_results["models"][mk]

        # Jailbreak — bootstrap on binary regex outcomes
        jb_binary = model_raw["jailbreak"]["asr_binary"]
        refusal_binary = model_raw["jailbreak"]["refusal_binary"]

        # Jailbreak — judge-based binary outcomes
        if mk in judge_verdicts:
            judge_binary = [1 if strongreject_is_success(v) else 0 for v in judge_verdicts[mk]]
        else:
            judge_binary = None

        # Sycophancy — bootstrap on binary outcomes
        syco_binary = model_raw["sycophancy"]["syco_binary"]
        clean_correct_binary = model_raw["sycophancy"]["clean_correct_binary"]
        wrapped_correct_binary = model_raw["sycophancy"]["wrapped_correct_binary"]

        # Capability — rollout CIs
        cap_ci: dict[str, dict] = {}
        cap_raw = model_raw.get("capabilities", {})
        for eval_name, metrics in cap_raw.items():
            cap_ci[eval_name] = {}
            for metric_key, stat in metrics.items():
                if isinstance(stat, dict) and "values" in stat:
                    cap_ci[eval_name][metric_key] = rollout_ci(stat["values"])
                elif isinstance(stat, (int, float)):
                    cap_ci[eval_name][metric_key] = {"mean": stat, "ci95_low": stat, "ci95_high": stat, "n": 1}

        entry = {
            "jailbreak": {
                "asr_regex": bootstrap_ci(jb_binary),
                "refusal_rate": bootstrap_ci(refusal_binary),
            },
            "sycophancy": {
                "sycophancy_rate": bootstrap_ci(syco_binary),
                "clean_accuracy": bootstrap_ci(clean_correct_binary),
                "wrapped_correct_rate": bootstrap_ci(wrapped_correct_binary),
            },
            "capabilities": cap_ci,
        }
        if judge_binary is not None:
            entry["jailbreak"]["asr_judge"] = bootstrap_ci(judge_binary)
            # Store per-row judge details
            entry["jailbreak"]["judge_details"] = [
                {"index": jb_rows[i]["index"], **judge_verdicts[mk][i]}
                for i in range(len(judge_verdicts[mk]))
            ]

        final[mk] = entry

    # --- Print summary table ---
    asr_key = "asr_judge" if any("asr_judge" in final[mk]["jailbreak"] for mk in model_keys) else "asr_regex"

    print("\n" + "=" * 90)
    print("FINAL EVALUATION RESULTS  (mean [95% CI])")
    print("=" * 90)
    print(f"  JB ASR via: {'StrongREJECT judge' if asr_key == 'asr_judge' else 'regex classifier'}")
    print(f"  N jailbreak={len(jb_rows)}, N sycophancy={len(sy_rows)}, bootstrap n={N_BOOTSTRAP}")
    print()

    col_w = 30
    headers = ["Metric"] + [mk.upper() for mk in model_keys]
    print(f"{'Metric':<{col_w}}" + "".join(f"  {h:>27}" for h in model_keys))
    print("-" * (col_w + 29 * len(model_keys)))

    def row(label, getter):
        vals = []
        for mk in model_keys:
            try:
                d = getter(final[mk])
                vals.append(fmt_ci(d))
            except (KeyError, TypeError):
                vals.append("N/A")
        print(f"{label:<{col_w}}" + "".join(f"  {v:>27}" for v in vals))

    row("JB ASR (wrapped) ↓",       lambda m: m["jailbreak"][asr_key])
    row("JB refusal rate (clean) ↑", lambda m: m["jailbreak"]["refusal_rate"])
    row("Syco rate (wrapped) ↓",    lambda m: m["sycophancy"]["sycophancy_rate"])
    row("Clean MCQ accuracy ↑",     lambda m: m["sycophancy"]["clean_accuracy"])
    row("Wrapped correct rate ↑",   lambda m: m["sycophancy"]["wrapped_correct_rate"])

    # Capability rows — use first model to discover eval/metric structure
    first_mk = model_keys[0]
    cap_rows_printed = False
    for eval_name, metrics in final[first_mk].get("capabilities", {}).items():
        for metric_key in metrics:
            if not cap_rows_printed:
                print()
                cap_rows_printed = True
            label = f"{eval_name} {metric_key.split(',')[0]}"
            row(label, lambda m, en=eval_name, mkey=metric_key: m["capabilities"][en][mkey])

    print("=" * 90)

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": final, "meta": {
            "n_jailbreak": len(jb_rows),
            "n_sycophancy": len(sy_rows),
            "n_bootstrap": N_BOOTSTRAP,
            "asr_key_used": asr_key,
        }}, f, indent=2)
    logger.info(f"Judged results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dump", default="results/final_eval_responses.jsonl",
                        help="Response dump JSONL from run_final_eval.py")
    parser.add_argument("--results", default="results/final_eval.json",
                        help="Raw results JSON from run_final_eval.py (contains binary outcomes + cap rollout scores)")
    parser.add_argument("--output", default="results/final_eval_judged.json",
                        help="Output path for judged results with CIs")
    parser.add_argument("--plot", default="results/final_eval_figure.pdf",
                        help="Output path for results plot PDF (default: results/final_eval_figure.pdf). Set '' to skip.")
    args = parser.parse_args()
    main(args)
