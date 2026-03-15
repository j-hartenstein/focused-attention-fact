"""Modal entrypoint for FACT experiments.

Usage:
    # Prepare data (runs on CPU, cheap)
    modal run modal_app/run_experiment.py::prepare_data

    # Phase 0: Baseline check
    modal run --detach modal_app/run_experiment.py::baseline --n-samples 100

    # Build patching dataset from strong-cue prompts (GPU, ~30 min for 1623 pairs)
    modal run --detach modal_app/run_experiment.py::build_patching_dataset

    # Phase 1: Attention patching — smoke test (20 per task, both variants)
    modal run --detach modal_app/run_experiment.py::patching --n-samples 20 --last-position-only
    modal run --detach modal_app/run_experiment.py::patching --n-samples 20
    # Phase 1: Attention patching — full run (both variants)
    modal run --detach modal_app/run_experiment.py::patching --last-position-only
    modal run --detach modal_app/run_experiment.py::patching

    # Phase 2: FACT fine-tuning (local smoke test)
    python experiments/run_fact.py --n-samples 20 --device cpu
    # Phase 2: FACT fine-tuning (on Modal) — no --n-samples uses full train set
    modal run --detach modal_app/run_experiment.py::fact_finetune
    # Optional: --lora-target-modules "q_proj k_proj v_proj o_proj gate_proj up_proj down_proj"

    # Phase 2 (CE-only baseline): same data, no FACT loss — for base vs CE vs FACT comparison
    modal run --detach modal_app/run_experiment.py::ce_finetune

    # Phase 2: Evaluate attention invariance (base vs FACT L_Inv plot)
    modal run --detach modal_app/run_experiment.py::eval_attention_invariance --n-samples 50

    # Behavioral validation: 100 val pairs per task (sub-sampled from 200-pair val set)
    modal run --detach modal_app/run_experiment.py::behavioral_validation --n-samples 100
    # 3-way comparison: base vs CE vs FACT (requires ce_finetune to have run first):
    modal run --detach modal_app/run_experiment.py::behavioral_validation --n-samples 100 --ce-adapter-path adapters/ce_lora
    # Skip capability evals (faster):
    modal run --detach modal_app/run_experiment.py::behavioral_validation --n-samples 100 --capability-evals ""
    # Requires data/capability_evals/ val sets — run locally first:
    #   python data/scripts/download_capability_evals.py

    # Generate jailbreak responses (Llama-3-8B, GPU)
    # Smoke test first (~20 pairs, quick sanity check):
    modal run --detach modal_app/run_experiment.py::smoke_jailbreak_responses
    # Full run (~1544 pairs, ~30 min):
    modal run --detach modal_app/run_experiment.py::generate_jailbreak_responses

    # Download responses then judge locally with StrongREJECT (requires OPENAI_API_KEY):
    modal volume get fact-data processed/jailbreak_responses.jsonl ./data/processed/
    python experiments/judge_jailbreaks.py --dump data/processed/jailbreak_responses.jsonl

    # Download results
    modal volume get fact-data results/ ./results/
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# In Modal container, entrypoint runs from /root/; project is mounted at /workspace
if "/workspace" not in sys.path:
    sys.path.insert(0, "/workspace")

import modal

from modal_app.app import (
    DATA_DIR,
    MODEL_CACHE_DIR,
    RESULTS_DIR,
    app,
    data_volume,
    image,
    model_volume,
)

PROJECT_ROOT = Path(__file__).parent.parent

# Modal 1.0+: use Image.add_local_dir instead of Mount (exclude large/irrelevant paths)
def _ignore_path(path) -> bool:
    # Modal may pass path as PosixPath; ensure we use a string for "in" checks.
    path_str = str(path)
    return any(
        part in path_str for part in [".git", "__pycache__", ".venv", "results", "data/raw", "data/processed"]
    )

workspace_image = image.add_local_dir(str(PROJECT_ROOT), "/workspace", ignore=_ignore_path)

GPU_CONFIG = "A100-80GB"
TIMEOUT = 15 * 60 * 60  # 1 hour max



@app.function(
    image=workspace_image,
    volumes={
        MODEL_CACHE_DIR: model_volume,
        DATA_DIR: data_volume,
    },
    timeout=TIMEOUT,
    cpu=4,
)
def prepare_data(
    n_jailbreaks: int = 1500,
    n_sycophancy: int = 1500,
    bct_path: str | None = None,
):
    """Download and prepare datasets (CPU-only, no GPU needed)."""
    import subprocess
    import sys

    sys.path.insert(0, "/workspace")

    # Prepare jailbreaks
    cmd = [
        sys.executable, "/workspace/data/scripts/prepare_jailbreaks.py",
        "--n-samples", str(n_jailbreaks),
    ]
    result = subprocess.run(cmd, cwd="/workspace", capture_output=False)
    if result.returncode != 0:
        raise RuntimeError("prepare_jailbreaks.py failed")

    # Prepare sycophancy
    cmd = [
        sys.executable, "/workspace/data/scripts/prepare_sycophancy.py",
        "--n-samples", str(n_sycophancy),
    ]
    if bct_path:
        cmd += ["--bct-path", bct_path]
    result = subprocess.run(cmd, cwd="/workspace", capture_output=False)
    if result.returncode != 0:
        raise RuntimeError("prepare_sycophancy.py failed")

    # Copy processed data to persistent volume
    import shutil
    import os
    os.makedirs(f"{DATA_DIR}/processed", exist_ok=True)
    for fname in ["jailbreaks.jsonl", "sycophancy.jsonl"]:
        src = f"/workspace/data/processed/{fname}"
        dst = f"{DATA_DIR}/processed/{fname}"
        if Path(src).exists():
            shutil.copy(src, dst)
            print(f"Copied {src} → {dst}")

    data_volume.commit()
    print("Data preparation complete.")


@app.function(
    image=workspace_image,
    gpu=GPU_CONFIG,
    volumes={
        MODEL_CACHE_DIR: model_volume,
        DATA_DIR: data_volume,
    },
    timeout=TIMEOUT,
)
def baseline(
    model: str = "meta-llama/Llama-3.2-3B-Instruct",
    n_samples: int = 100,
    judge: bool = False,
):
    """Phase 0: Compute baseline ASR and sycophancy rates."""
    import subprocess
    import sys
    import shutil
    import os

    sys.path.insert(0, "/workspace")

    # Symlink data from volume into expected path
    os.makedirs("/workspace/data/processed", exist_ok=True)
    for fname in ["jailbreaks.jsonl", "sycophancy.jsonl"]:
        src = f"{DATA_DIR}/processed/{fname}"
        dst = f"/workspace/data/processed/{fname}"
        if Path(src).exists() and not Path(dst).exists():
            shutil.copy(src, dst)

    os.makedirs("/workspace/results", exist_ok=True)

    cmd = [
        sys.executable, "/workspace/experiments/run_baseline.py",
        "--model", model,
        "--n-samples", str(n_samples),
        "--device", "cuda",
        "--output", "/workspace/results/baseline.json",
    ]
    if judge:
        cmd.append("--judge")

    result = subprocess.run(cmd, cwd="/workspace")
    if result.returncode != 0:
        raise RuntimeError("run_baseline.py failed")

    # Copy results to volume
    os.makedirs(f"{DATA_DIR}/results", exist_ok=True)
    shutil.copy("/workspace/results/baseline.json", f"{DATA_DIR}/results/baseline.json")
    data_volume.commit()
    print(f"Baseline results saved to {DATA_DIR}/results/baseline.json")


@app.function(
    image=workspace_image,
    gpu=GPU_CONFIG,
    volumes={
        MODEL_CACHE_DIR: model_volume,
        DATA_DIR: data_volume,
    },
    timeout=TIMEOUT,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
    ],
)
def patching(
    model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    n_samples: int = 0,
    tasks: str = "",
    attacks: str = "",
    conditions: str = "",
    batch_size: int = 16,
    max_length: int = 1024,
    max_new_tokens: int = 100,
    last_position_only: bool = False,
    last_position_noncore_mode: str = "zero",
    last_position_noncore_epsilon: float = 1e-8,
    last_position_noncore_beta: float = 0.01,
    full_matrix_noncore_mode: str = "zero",
    full_matrix_noncore_beta: float = 0.01,
    output: str = "",
):
    """Phase 1: Run attention patching sweep using pre-filtered informative pairs.

    Output filename defaults to patching_last_pos.json or patching_full.json
    depending on last_position_only, so both runs can coexist in the volume.
    """
    import subprocess
    import sys
    import shutil
    import os
    import time

    sys.path.insert(0, "/workspace")

    if not output:
        output = "patching_last_pos.json" if last_position_only else "patching_full.json"

    # data/patching/ is included via add_local_dir (not excluded by _ignore_path)
    os.makedirs("/workspace/results", exist_ok=True)
    out_path = f"/workspace/results/{output}"

    cmd = [
        sys.executable, "/workspace/experiments/run_patching.py",
        "--model", model,
        "--device", "cuda",
        "--max-length", str(max_length),
        "--max-new-tokens", str(max_new_tokens),
        "--batch-size", str(batch_size),
        "--output", out_path,
    ]
    if n_samples > 0:
        cmd += ["--n-samples", str(n_samples)]
    if tasks:
        cmd += ["--tasks"] + tasks.split()
    if attacks:
        cmd += ["--attacks"] + attacks.split()
    if conditions:
        cmd += ["--conditions"] + conditions.split()
    if last_position_only:
        cmd += ["--last-position-only"]
        cmd += ["--last-position-noncore-mode", last_position_noncore_mode]
        if last_position_noncore_mode == "epsilon":
            cmd += ["--last-position-noncore-epsilon", str(last_position_noncore_epsilon)]
        if last_position_noncore_mode == "beta":
            cmd += ["--last-position-noncore-beta", str(last_position_noncore_beta)]
    else:
        cmd += ["--full-matrix-noncore-mode", full_matrix_noncore_mode]
        if full_matrix_noncore_mode == "beta":
            cmd += ["--full-matrix-noncore-beta", str(full_matrix_noncore_beta)]

    print(f"Running: {' '.join(cmd)}")
    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd="/workspace")
    elapsed = time.perf_counter() - t0
    print(f"Patching finished in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    if result.returncode != 0:
        raise RuntimeError("run_patching.py failed")

    os.makedirs(f"{DATA_DIR}/results", exist_ok=True)
    shutil.copy(out_path, f"{DATA_DIR}/results/{output}")
    data_volume.commit()
    print(f"Patching results saved to {DATA_DIR}/results/{output}")


@app.function(
    image=workspace_image,
    gpu=GPU_CONFIG,
    volumes={
        MODEL_CACHE_DIR: model_volume,
        DATA_DIR: data_volume,
    },
    timeout=TIMEOUT,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
    ],
)
def fact_finetune(
    model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    n_samples: int = 0,
    tasks: str = "",
    attacks: str = "",
    max_length: int = 1024,
    batch_size: int = 2,
    epochs: int = 3,
    lr: float = 1e-4,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: str = "q_proj k_proj v_proj o_proj gate_proj up_proj down_proj",
    adapter_output: str = "",
    zero_clean_noncore: bool = False,
    renormalize_clean_masked: bool = False,
    fact_loss_weight: float = 0.25,
    max_grad_norm: float = 1.0,
    fact_skip_layers: str = "0 1 2 3 4 5",
    if_share: float = 0.3,
    layer_eval_batches: int = 0,
):
    """Phase 2: FACT fine-tuning with LoRA. Uses all-responses data with informative filtering.

    Saves adapter to the data volume (default: adapters/fact_lora).
    """
    import subprocess
    import sys
    import shutil
    import os
    import time

    sys.path.insert(0, "/workspace")

    # Copy FACT data files from volume (data/processed is excluded from image mount)
    os.makedirs("/workspace/data/processed", exist_ok=True)
    for fname in ["jailbreak_all_responses.jsonl", "sycophancy_strong_cues_all_responses.jsonl", "if_dataset.jsonl"]:
        src = f"{DATA_DIR}/processed/{fname}"
        dst = f"/workspace/data/processed/{fname}"
        if Path(src).exists() and not Path(dst).exists():
            shutil.copy(src, dst)

    if not adapter_output:
        adapter_output = "adapters/fact_lora"
    out_dir = f"/workspace/results/{adapter_output}"
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        sys.executable, "/workspace/experiments/run_fact.py",
        "--model", model,
        "--device", "cuda",
        "--max-length", str(max_length),
        "--batch-size", str(batch_size),
        "--epochs", str(epochs),
        "--lr", str(lr),
        "--lora-r", str(lora_r),
        "--lora-alpha", str(lora_alpha),
        "--lora-dropout", str(lora_dropout),
        "--adapter-output", out_dir,
    ]
    if lora_target_modules:
        cmd += ["--lora-target-modules"] + lora_target_modules.split()
    if n_samples > 0:
        cmd += ["--n-samples", str(n_samples)]
    if tasks:
        cmd += ["--tasks"] + tasks.split()
    if attacks:
        cmd += ["--attacks"] + attacks.split()
    if zero_clean_noncore:
        cmd.append("--zero-clean-noncore")
    if renormalize_clean_masked:
        cmd.append("--renormalize-clean-masked")
    cmd += ["--fact-loss-weight", str(fact_loss_weight)]
    cmd += ["--max-grad-norm", str(max_grad_norm)]
    if fact_skip_layers:
        cmd += ["--fact-skip-layers"] + fact_skip_layers.split()
    cmd += ["--if-share", str(if_share)]
    if layer_eval_batches > 0:
        cmd += ["--layer-eval-batches", str(layer_eval_batches)]

    print(f"Running: {' '.join(cmd)}")
    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd="/workspace")
    elapsed = time.perf_counter() - t0
    print(f"FACT fine-tuning finished in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    if result.returncode != 0:
        raise RuntimeError("run_fact.py failed")

    # Copy adapter dir to volume
    dest_dir = f"{DATA_DIR}/{adapter_output}"
    os.makedirs(dest_dir, exist_ok=True)
    for name in os.listdir(out_dir):
        src = os.path.join(out_dir, name)
        dst = os.path.join(dest_dir, name)
        if os.path.isfile(src):
            shutil.copy(src, dst)
        else:
            shutil.copytree(src, dst, dirs_exist_ok=True)
    data_volume.commit()
    print(f"Adapter saved to {dest_dir}")


@app.function(
    image=workspace_image,
    gpu=GPU_CONFIG,
    volumes={
        MODEL_CACHE_DIR: model_volume,
        DATA_DIR: data_volume,
    },
    timeout=TIMEOUT,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
    ],
)
def ce_finetune(
    model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    n_samples: int = 0,
    tasks: str = "",
    attacks: str = "",
    max_length: int = 1024,
    batch_size: int = 2,
    epochs: int = 3,
    lr: float = 2e-5,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: str = "q_proj k_proj v_proj o_proj gate_proj up_proj down_proj",
    adapter_output: str = "",
    max_grad_norm: float = 1.0,
    if_share: float = 0.3,
):
    """CE-only baseline fine-tuning with LoRA (no FACT loss).

    Identical to fact_finetune but with fact_loss_weight=0, so only cross-entropy
    on the clean response is optimised. Use this to get a CE-only baseline adapter
    for base vs CE vs FACT comparisons.

    Saves adapter to the data volume (default: adapters/ce_lora).
    """
    import subprocess
    import sys
    import shutil
    import os
    import time

    sys.path.insert(0, "/workspace")

    # Copy FACT data files from volume (data/processed is excluded from image mount)
    os.makedirs("/workspace/data/processed", exist_ok=True)
    for fname in ["jailbreak_all_responses.jsonl", "sycophancy_strong_cues_all_responses.jsonl", "if_dataset.jsonl"]:
        src = f"{DATA_DIR}/processed/{fname}"
        dst = f"/workspace/data/processed/{fname}"
        if Path(src).exists() and not Path(dst).exists():
            shutil.copy(src, dst)

    if not adapter_output:
        adapter_output = "adapters/ce_lora"
    out_dir = f"/workspace/results/{adapter_output}"
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        sys.executable, "/workspace/experiments/run_fact.py",
        "--model", model,
        "--device", "cuda",
        "--max-length", str(max_length),
        "--batch-size", str(batch_size),
        "--epochs", str(epochs),
        "--lr", str(lr),
        "--lora-r", str(lora_r),
        "--lora-alpha", str(lora_alpha),
        "--lora-dropout", str(lora_dropout),
        "--adapter-output", out_dir,
        "--fact-loss-weight", "0",  # CE-only: no FACT invariance loss
    ]
    if lora_target_modules:
        cmd += ["--lora-target-modules"] + lora_target_modules.split()
    if n_samples > 0:
        cmd += ["--n-samples", str(n_samples)]
    if tasks:
        cmd += ["--tasks"] + tasks.split()
    if attacks:
        cmd += ["--attacks"] + attacks.split()
    cmd += ["--max-grad-norm", str(max_grad_norm)]
    cmd += ["--if-share", str(if_share)]

    print(f"Running: {' '.join(cmd)}")
    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd="/workspace")
    elapsed = time.perf_counter() - t0
    print(f"CE fine-tuning finished in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    if result.returncode != 0:
        raise RuntimeError("run_fact.py (CE-only) failed")

    # Copy adapter dir to volume
    dest_dir = f"{DATA_DIR}/{adapter_output}"
    os.makedirs(dest_dir, exist_ok=True)
    for name in os.listdir(out_dir):
        src = os.path.join(out_dir, name)
        dst = os.path.join(dest_dir, name)
        if os.path.isfile(src):
            shutil.copy(src, dst)
        else:
            shutil.copytree(src, dst, dirs_exist_ok=True)
    data_volume.commit()
    print(f"CE adapter saved to {dest_dir}")


@app.function(
    image=workspace_image,
    gpu=GPU_CONFIG,
    volumes={
        MODEL_CACHE_DIR: model_volume,
        DATA_DIR: data_volume,
    },
    timeout=TIMEOUT,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
    ],
)
def eval_layer_loss(
    adapter_paths: str = "adapters/fact_lora_3ep_1e4lr_flw025",
    output_names: str = "layer_history_skiplayers.json",
    base_model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    n_batches: int = 50,
    batch_size: int = 4,
):
    """Post-hoc per-layer L_Inv eval for one or more trained adapters.

    Runs eval_layer_loss.py on GPU, saves layer_history JSON(s) to the data volume.

    Args:
        adapter_paths: Space-separated adapter paths relative to DATA_DIR.
        output_names:  Space-separated output filenames (saved to DATA_DIR/results/).
    """
    import subprocess
    import sys
    import os
    import shutil
    import time

    sys.path.insert(0, "/workspace")

    adapter_list = adapter_paths.split()
    output_list  = output_names.split()
    if len(adapter_list) != len(output_list):
        raise ValueError("adapter_paths and output_names must have the same number of entries")

    full_adapter_paths = [f"{DATA_DIR}/{p}" for p in adapter_list]
    full_output_paths  = [f"/workspace/results/{n}" for n in output_list]

    for p in full_adapter_paths:
        if not Path(p).exists():
            raise FileNotFoundError(f"Adapter not found: {p}")

    os.makedirs("/workspace/results", exist_ok=True)
    os.makedirs("/workspace/data/processed", exist_ok=True)
    # Copy processed data needed for val split
    for fname in ["jailbreak_all_responses.jsonl", "sycophancy_strong_cues_all_responses.jsonl"]:
        src = f"{DATA_DIR}/processed/{fname}"
        dst = f"/workspace/data/processed/{fname}"
        if Path(src).exists() and not Path(dst).exists():
            shutil.copy(src, dst)

    cmd = [
        sys.executable, "/workspace/experiments/eval_layer_loss.py",
        "--adapter-path", *full_adapter_paths,
        "--output",       *full_output_paths,
        "--base-model",   base_model,
        "--device",       "cuda",
        "--n-batches",    str(n_batches),
        "--batch-size",   str(batch_size),
    ]
    print(f"Running: {' '.join(cmd)}")
    t0 = time.perf_counter()
    env = os.environ.copy()
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    result = subprocess.run(cmd, cwd="/workspace", env=env)
    elapsed = time.perf_counter() - t0
    print(f"eval_layer_loss finished in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    if result.returncode != 0:
        raise RuntimeError("eval_layer_loss.py failed (exit code %d)" % result.returncode)

    os.makedirs(f"{DATA_DIR}/results", exist_ok=True)
    for local_path, name in zip(full_output_paths, output_list):
        if Path(local_path).exists():
            shutil.copy(local_path, f"{DATA_DIR}/results/{name}")
            print(f"Saved {DATA_DIR}/results/{name}")
    data_volume.commit()


@app.function(
    image=workspace_image,
    gpu=GPU_CONFIG,
    volumes={
        MODEL_CACHE_DIR: model_volume,
        DATA_DIR: data_volume,
    },
    timeout=TIMEOUT,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
    ],
)
def eval_attention_invariance(
    adapter_path: str = "adapters/fact_lora",
    base_model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    n_samples: int = 50,
    tasks: str = "",
    max_length: int = 1024,
    batch_size: int = 1,
    output: str = "results/l_inv_eval.png",
):
    """Evaluate L_Inv: base vs base+FACT. Requires adapter already on volume (run fact_finetune first).

    Saves plot to the data volume (default: results/l_inv_eval.png).
    """
    import subprocess
    import sys
    import os
    import shutil
    import time

    sys.path.insert(0, "/workspace")

    # Adapter lives on the volume
    adapter_full = f"{DATA_DIR}/{adapter_path}"
    if not Path(adapter_full).exists():
        raise FileNotFoundError(
            f"Adapter not found at {adapter_full}. Run fact_finetune first."
        )

    out_path = f"/workspace/{output}"
    os.makedirs(Path(out_path).parent, exist_ok=True)

    cmd = [
        sys.executable, "/workspace/experiments/eval_attention_invariance.py",
        "--adapter-path", adapter_full,
        "--base-model", base_model,
        "--n-samples", str(n_samples),
        "--max-length", str(max_length),
        "--batch-size", str(batch_size),
        "--device", "cuda",
        "--output", out_path,
    ]
    if tasks:
        cmd += ["--tasks"] + tasks.split()

    print(f"Running: {' '.join(cmd)}")
    t0 = time.perf_counter()
    # Reduce CUDA fragmentation during long attention-heavy eval runs.
    env = os.environ.copy()
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    result = subprocess.run(cmd, cwd="/workspace", env=env)
    elapsed = time.perf_counter() - t0
    print(f"Eval finished in {elapsed:.1f}s")

    if result.returncode != 0:
        raise RuntimeError(
            "eval_attention_invariance.py failed (exit code %d). "
            "Check logs above for the root cause (e.g. CUDA OOM)." % result.returncode
        )

    os.makedirs(f"{DATA_DIR}/results", exist_ok=True)
    shutil.copy(out_path, f"{DATA_DIR}/{output}")
    data_volume.commit()
    print(f"Plot saved to {DATA_DIR}/{output}")


@app.function(
    image=workspace_image,
    gpu=GPU_CONFIG,
    volumes={
        MODEL_CACHE_DIR: model_volume,
        DATA_DIR: data_volume,
    },
    timeout=TIMEOUT,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
    ],
)
def behavioral_validation(
    adapter_path: str = "adapters/fact_lora",
    ce_adapter_path: str = "",
    model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    n_samples: int = 100,
    max_new_tokens: int = 100,
    jailbreak_data: str = "data/processed/jailbreak_all_responses.jsonl",
    sycophancy_data: str = "data/processed/sycophancy_strong_cues_all_responses.jsonl",
    output: str = "results/behavioral_validation.json",
    capability_evals: str = "ifeval mmlu",
):
    """Compare base vs base+CE vs base+FACT on behavioral metrics and capability benchmarks.

    Requires FACT adapter on the data volume (run fact_finetune first). Optionally pass
    ce_adapter_path (from ce_finetune) for a 3-way comparison: base / CE / FACT.

    Uses all_responses files with informative_only=True so base sycophancy/ASR rates reflect
    the hard cases where the base model was actually fooled.

    Capability evals use static val sets from data/capability_evals/ (committed to the repo).
    Pass capability_evals="" to skip them.
    """
    import subprocess
    import sys
    import os
    import shutil
    import time

    sys.path.insert(0, "/workspace")

    adapter_full = f"{DATA_DIR}/{adapter_path}"
    if not Path(adapter_full).exists():
        raise FileNotFoundError(
            f"Adapter not found at {adapter_full}. Run fact_finetune first."
        )

    ce_adapter_full = ""
    if ce_adapter_path:
        ce_adapter_full = f"{DATA_DIR}/{ce_adapter_path}"
        if not Path(ce_adapter_full).exists():
            raise FileNotFoundError(
                f"CE adapter not found at {ce_adapter_full}. Run ce_finetune first."
            )

    # Copy processed data from volume so script finds it
    os.makedirs("/workspace/data/processed", exist_ok=True)
    for fname in [
        Path(jailbreak_data).name,
        Path(sycophancy_data).name,
    ]:
        src = f"{DATA_DIR}/processed/{fname}"
        dst = f"/workspace/data/processed/{fname}"
        if Path(src).exists() and not Path(dst).exists():
            shutil.copy(src, dst)
    jailbreak_arg = f"/workspace/data/processed/{Path(jailbreak_data).name}"
    sycophancy_arg = f"/workspace/data/processed/{Path(sycophancy_data).name}"

    os.makedirs("/workspace/results", exist_ok=True)
    out_path = f"/workspace/{output}"
    os.makedirs(Path(out_path).parent, exist_ok=True)
    # Dump path: same dir as output, stem_responses.jsonl (for inspecting base vs FACT raw strings)
    dump_path = str(Path(out_path).with_stem(Path(out_path).stem + "_responses").with_suffix(".jsonl"))

    cmd = [
        sys.executable, "/workspace/experiments/run_behavioral_validation.py",
        "--model", model,
        "--adapter-path", adapter_full,
        "--n-samples", str(n_samples),
        "--max-new-tokens", str(max_new_tokens),
        "--device", "cuda",
        "--jailbreak-data", jailbreak_arg,
        "--sycophancy-data", sycophancy_arg,
        "--output", out_path,
        "--dump", dump_path,
        "--capability-evals", capability_evals,
        "--capability-val-dir", "/workspace/data/capability_evals",
    ]
    if ce_adapter_full:
        cmd += ["--ce-adapter-path", ce_adapter_full]

    print(f"Running: {' '.join(cmd)}")
    t0 = time.perf_counter()
    # Reduce CUDA fragmentation (helps when loading base then base+adapter sequentially)
    env = os.environ.copy()
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    result = subprocess.run(cmd, cwd="/workspace", env=env)
    elapsed = time.perf_counter() - t0
    print(f"Behavioral validation finished in {elapsed:.1f}s")

    if result.returncode != 0:
        raise RuntimeError(
            "run_behavioral_validation.py failed (exit code %d). "
            "Check logs above for the actual error (e.g. OOM, missing file)." % result.returncode
        )

    os.makedirs(f"{DATA_DIR}/results", exist_ok=True)
    shutil.copy(out_path, f"{DATA_DIR}/{output}")
    if Path(dump_path).exists():
        shutil.copy(dump_path, f"{DATA_DIR}/results/{Path(dump_path).name}")
        print(f"Raw responses dump saved to {DATA_DIR}/results/{Path(dump_path).name}")
    data_volume.commit()
    print(f"Results saved to {DATA_DIR}/{output}")


@app.function(
    image=workspace_image,
    gpu=GPU_CONFIG,
    volumes={
        MODEL_CACHE_DIR: model_volume,
        DATA_DIR: data_volume,
    },
    timeout=TIMEOUT,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
    ],
)
def final_eval(
    adapter_path: str = "adapters/fact_lora_3ep_1e4lr_flw025",
    ce_adapter_path: str = "adapters/ce_lora_3ep_1e4lr",
    model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    n_samples: int = 0,
    max_new_tokens: int = 100,
    capability_evals: str = "ifeval mmlu",
    cap_rollouts: int = 5,
    cap_samples_per_rollout: int = 200,
    output: str = "results/final_eval.json",
    dump: str = "results/final_eval_responses.jsonl",
):
    """Final evaluation on held-out test sets with bootstrap CIs.

    Runs generation for base / CE / FACT on jailbreak + sycophancy test sets,
    then runs capability evals with multiple independent random rollouts.
    Saves response dump and raw results to the data volume for local judging.

    Args:
        n_samples: Cap on test samples per task (0 = all 200). Use 5 for smoke test.
        cap_rollouts: Independent capability eval rollouts (default 5).
        cap_samples_per_rollout: Samples per capability rollout (default 200).
    """
    import subprocess
    import sys
    import os
    import shutil
    import time

    sys.path.insert(0, "/workspace")

    adapter_full = f"{DATA_DIR}/{adapter_path}"
    if not Path(adapter_full).exists():
        raise FileNotFoundError(f"FACT adapter not found: {adapter_full}")

    ce_adapter_full = ""
    if ce_adapter_path:
        ce_adapter_full = f"{DATA_DIR}/{ce_adapter_path}"
        if not Path(ce_adapter_full).exists():
            raise FileNotFoundError(f"CE adapter not found: {ce_adapter_full}")

    # Copy test data from volume to workspace
    os.makedirs("/workspace/data/processed/test", exist_ok=True)
    for fname in ["jailbreaks.jsonl", "sycophancy_non_cot_strong.jsonl"]:
        src = f"{DATA_DIR}/processed/test/{fname}"
        dst = f"/workspace/data/processed/test/{fname}"
        if Path(src).exists() and not Path(dst).exists():
            shutil.copy(src, dst)
        elif not Path(src).exists():
            raise FileNotFoundError(f"Test data not found in volume: {src}")

    os.makedirs("/workspace/results", exist_ok=True)
    out_path = f"/workspace/{output}"
    dump_path = f"/workspace/{dump}"
    os.makedirs(Path(out_path).parent, exist_ok=True)

    cmd = [
        sys.executable, "/workspace/experiments/run_final_eval.py",
        "--model", model,
        "--adapter-path", adapter_full,
        "--jailbreak-data", "/workspace/data/processed/test/jailbreaks.jsonl",
        "--sycophancy-data", "/workspace/data/processed/test/sycophancy_non_cot_strong.jsonl",
        "--max-new-tokens", str(max_new_tokens),
        "--device", "cuda",
        "--capability-evals", capability_evals,
        "--cap-rollouts", str(cap_rollouts),
        "--cap-samples-per-rollout", str(cap_samples_per_rollout),
        "--output", out_path,
        "--dump", dump_path,
    ]
    if ce_adapter_full:
        cmd += ["--ce-adapter-path", ce_adapter_full]
    if n_samples and n_samples > 0:
        cmd += ["--n-samples", str(n_samples)]

    print(f"Running: {' '.join(cmd)}")
    t0 = time.perf_counter()
    env = os.environ.copy()
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    result = subprocess.run(cmd, cwd="/workspace", env=env)
    elapsed = time.perf_counter() - t0
    print(f"final_eval finished in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    if result.returncode != 0:
        raise RuntimeError("run_final_eval.py failed (exit code %d)" % result.returncode)

    os.makedirs(f"{DATA_DIR}/results", exist_ok=True)
    for local_path, vol_name in [(out_path, output), (dump_path, dump)]:
        if Path(local_path).exists():
            shutil.copy(local_path, f"{DATA_DIR}/{vol_name}")
            print(f"Saved {DATA_DIR}/{vol_name}")
    data_volume.commit()


@app.function(
    image=workspace_image,
    gpu=GPU_CONFIG,
    volumes={
        MODEL_CACHE_DIR: model_volume,
        DATA_DIR: data_volume,
    },
    timeout=TIMEOUT,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
    ],
)
def sycophancy_only(
    adapter_path: str = "adapters/fact_lora_3ep_1e4lr_flw025",
    ce_adapter_path: str = "adapters/ce_lora_3ep_1e4lr",
    model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    n_samples: int = 0,
    max_new_tokens: int = 100,
    sycophancy_data: str = "data/processed/test/sycophancy_non_cot_strong.jsonl",
    results: str = "results/final_eval_strong.json",
    dump: str = "results/final_eval_responses_strong.jsonl",
):
    """Re-run only the sycophancy eval and patch results/dump files in-place.

    Reads jailbreak + capability data from the existing final_eval.json on the
    volume and writes a new file so the original is preserved.
    After this runs, download and re-judge + re-plot locally:

        modal volume get fact-data results/final_eval_strong.json ./results/
        python experiments/patch_sycophancy_judged.py
        python experiments/plot_final_eval.py --results results/final_eval_judged_strong.json --output results/final_eval_figure_strong.pdf

    Usage:
        modal run --detach modal_app/run_experiment.py::sycophancy_only
    """
    import subprocess
    import sys
    import os
    import shutil
    import time

    sys.path.insert(0, "/workspace")

    adapter_full = f"{DATA_DIR}/{adapter_path}"
    if not Path(adapter_full).exists():
        raise FileNotFoundError(f"FACT adapter not found: {adapter_full}")

    ce_adapter_full = ""
    if ce_adapter_path:
        ce_adapter_full = f"{DATA_DIR}/{ce_adapter_path}"
        if not Path(ce_adapter_full).exists():
            raise FileNotFoundError(f"CE adapter not found: {ce_adapter_full}")

    # Copy test data and existing results from volume to workspace
    os.makedirs("/workspace/data/processed/test", exist_ok=True)
    os.makedirs("/workspace/results", exist_ok=True)

    syco_fname = Path(sycophancy_data).name
    src = f"{DATA_DIR}/processed/test/{syco_fname}"
    dst = f"/workspace/data/processed/test/{syco_fname}"
    if Path(src).exists():
        shutil.copy(src, dst)
    else:
        # Fall back to workspace-mounted version (included via add_local_dir)
        if not Path(f"/workspace/{sycophancy_data}").exists():
            raise FileNotFoundError(f"Sycophancy test data not found: {src}")
        dst = f"/workspace/{sycophancy_data}"

    # Copy source results + dump from volume (originals, read-only)
    src_results = "results/final_eval.json"
    src_dump    = "results/final_eval_responses.jsonl"
    for vol_rel, local_rel in [
        (src_results, f"/workspace/{src_results}"),
        (src_dump,    f"/workspace/{src_dump}"),
    ]:
        vol_path = f"{DATA_DIR}/{vol_rel}"
        if Path(vol_path).exists():
            os.makedirs(Path(f"/workspace/{vol_rel}").parent, exist_ok=True)
            shutil.copy(vol_path, local_rel)
        else:
            raise FileNotFoundError(
                f"Existing results file not found in volume: {vol_path}. "
                "Run final_eval first."
            )

    out_path  = f"/workspace/{results}"
    dump_path = f"/workspace/{dump}"
    os.makedirs(Path(out_path).parent, exist_ok=True)

    cmd = [
        sys.executable, "/workspace/experiments/run_sycophancy_only.py",
        "--model", model,
        "--adapter-path", adapter_full,
        "--sycophancy-data", dst,
        "--max-new-tokens", str(max_new_tokens),
        "--device", "cuda",
        "--source-results", f"/workspace/{src_results}",
        "--source-dump",    f"/workspace/{src_dump}",
        "--results", out_path,
        "--dump", dump_path,
    ]
    if ce_adapter_full:
        cmd += ["--ce-adapter-path", ce_adapter_full]
    if n_samples and n_samples > 0:
        cmd += ["--n-samples", str(n_samples)]

    print(f"Running: {' '.join(cmd)}")
    t0 = time.perf_counter()
    env = os.environ.copy()
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    result = subprocess.run(cmd, cwd="/workspace", env=env)
    elapsed = time.perf_counter() - t0
    print(f"sycophancy_only finished in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    if result.returncode != 0:
        raise RuntimeError("run_sycophancy_only.py failed (exit code %d)" % result.returncode)

    # Write back patched files to volume
    os.makedirs(f"{DATA_DIR}/results", exist_ok=True)
    for local_path, vol_rel in [(out_path, results), (dump_path, dump)]:
        if Path(local_path).exists():
            shutil.copy(local_path, f"{DATA_DIR}/{vol_rel}")
            print(f"Saved {DATA_DIR}/{vol_rel}")
    data_volume.commit()


@app.function(
    image=workspace_image,
    gpu=GPU_CONFIG,
    volumes={
        MODEL_CACHE_DIR: model_volume,
        DATA_DIR: data_volume,
    },
    timeout=TIMEOUT,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
    ],
)
def build_patching_dataset(
    model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    input_file: str = "sycophancy_strong_cues.jsonl",
    n_informative: int = 170,
    batch_size: int = 64,
    max_new_tokens: int = 60,
    seed: int = 42,
):
    """Generate responses for all strong-cue pairs and build the patching dataset.

    Step 1: Run filter_correct.py on sycophancy_strong_cues.jsonl to get clean +
            wrapped responses for every pair.
    Step 2: Run sample_informative.py to select n_informative pairs where the model
            was correct on clean but sycophantic on wrapped.

    Outputs saved to the data volume:
        processed/sycophancy_strong_cues_all_responses.jsonl   all pairs + responses
        processed/sycophancy_strong_cues_informative.jsonl     ~n_informative pairs for patching
        results/build_patching_dataset_stats.json              counts + rates
    """
    import subprocess
    import sys
    import shutil
    import os

    sys.path.insert(0, "/workspace")

    os.makedirs("/workspace/data/processed", exist_ok=True)
    os.makedirs("/workspace/results", exist_ok=True)

    # Copy input data from volume
    src = f"{DATA_DIR}/processed/{input_file}"
    dst = f"/workspace/data/processed/{input_file}"
    if Path(src).exists() and not Path(dst).exists():
        shutil.copy(src, dst)

    stem = Path(input_file).stem
    all_responses = f"data/processed/{stem}_all_responses.jsonl"
    informative   = f"data/processed/{stem}_informative.jsonl"
    stats_file    = f"results/build_patching_dataset_stats.json"

    # Step 1: generate clean + wrapped responses and classify
    cmd1 = [
        sys.executable, "/workspace/data/scripts/filter_correct.py",
        "--model",          model,
        "--input",          f"data/processed/{input_file}",
        "--output",         f"data/processed/{stem}_filtered.jsonl",  # intermediate, unused
        "--all-responses",  all_responses,
        "--stats",          stats_file,
        "--batch-size",     str(batch_size),
        "--max-new-tokens", str(max_new_tokens),
        "--device",         "cuda",
    ]
    print(f"Step 1: {' '.join(cmd1)}")
    r1 = subprocess.run(cmd1, cwd="/workspace")
    if r1.returncode != 0:
        raise RuntimeError("filter_correct.py failed")

    # Step 2: sample informative pairs
    cmd2 = [
        sys.executable, "/workspace/data/scripts/sample_informative.py",
        "--input",  all_responses,
        "--output", informative,
        "--n",      str(n_informative),
        "--seed",   str(seed),
    ]
    print(f"Step 2: {' '.join(cmd2)}")
    r2 = subprocess.run(cmd2, cwd="/workspace")
    if r2.returncode != 0:
        raise RuntimeError("sample_informative.py failed")

    # Copy outputs to volume
    os.makedirs(f"{DATA_DIR}/processed", exist_ok=True)
    os.makedirs(f"{DATA_DIR}/results", exist_ok=True)
    for local_rel, remote in [
        (all_responses, f"{DATA_DIR}/processed/{Path(all_responses).name}"),
        (informative,   f"{DATA_DIR}/processed/{Path(informative).name}"),
        (stats_file,    f"{DATA_DIR}/results/{Path(stats_file).name}"),
    ]:
        local_path = Path(f"/workspace/{local_rel}")
        if local_path.exists():
            shutil.copy(local_path, remote)
            print(f"Saved {remote}")

    data_volume.commit()
    print("Done.")


@app.function(
    image=workspace_image,
    gpu=GPU_CONFIG,
    volumes={
        MODEL_CACHE_DIR: model_volume,
        DATA_DIR: data_volume,
    },
    timeout=TIMEOUT,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
    ],
)
def generate_jailbreak_responses(
    model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    input_file: str = "jailbreaks.jsonl",
    batch_size: int = 32,
    max_new_tokens: int = 150,
    n_samples: int = 0,
    output: str = "jailbreak_responses.jsonl",
):
    """Generate clean + wrapped responses for every jailbreak pair using Llama-3-8B.

    Saves a JSONL dump to the data volume at processed/<output> with fields:
        prompt_idx, attack, condition (clean|wrapped), prompt, response

    Args:
        model:          HuggingFace model ID (default Meta-Llama-3-8B-Instruct).
        input_file:     JSONL filename in DATA_DIR/processed/ (default jailbreaks.jsonl).
        batch_size:     Generation batch size (default 32).
        max_new_tokens: Max tokens to generate per response (default 150).
        n_samples:      Cap number of pairs (0 = all).
        output:         Output filename saved to DATA_DIR/processed/.

    Cost estimate (A100-40GB): ~1544 pairs × 2 conditions = 3088 forward passes.
    At batch_size=32 and ~200ms/batch → ~20s/batch → ~30 min total.
    """
    import json
    import os
    import shutil
    import sys
    import time

    sys.path.insert(0, "/workspace")

    from fact.model import load_model_and_tokenizer
    from fact.utils import get_logger

    logger = get_logger("generate_jailbreak_responses")

    # Copy input data from volume if not already present
    os.makedirs("/workspace/data/processed", exist_ok=True)
    src = f"{DATA_DIR}/processed/{input_file}"
    dst = f"/workspace/data/processed/{input_file}"
    if Path(src).exists() and not Path(dst).exists():
        shutil.copy(src, dst)

    # Load pairs
    pairs = []
    with open(dst) as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    if n_samples > 0:
        pairs = pairs[:n_samples]
    logger.info(f"Loaded {len(pairs)} jailbreak pairs from {dst}")

    # Load model
    model_obj, tokenizer = load_model_and_tokenizer(model_name=model, device="cuda")

    import torch

    def _batch_generate(prompts: list[str]) -> list[str]:
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                           max_length=1024).to(model_obj.device)
        with torch.no_grad():
            out = model_obj.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        responses = []
        for i, seq in enumerate(out):
            # Strip the input tokens, decode only the generated portion
            input_len = inputs["input_ids"].shape[1]
            gen_tokens = seq[input_len:]
            responses.append(tokenizer.decode(gen_tokens, skip_special_tokens=True))
        return responses

    dump_rows = []
    conditions = [("clean", "clean"), ("wrapped", "wrapped")]

    t0 = time.perf_counter()
    for cond_key, cond_label in conditions:
        prompts_all = [p[cond_key] for p in pairs]
        logger.info(f"Generating {cond_label} responses ({len(prompts_all)} prompts, batch_size={batch_size})…")
        responses_all: list[str] = []
        for start in range(0, len(prompts_all), batch_size):
            batch = prompts_all[start:start + batch_size]
            responses_all.extend(_batch_generate(batch))
            if (start // batch_size) % 10 == 0:
                logger.info(f"  {len(responses_all)}/{len(prompts_all)} done")
        for i, (pair, resp) in enumerate(zip(pairs, responses_all)):
            dump_rows.append({
                "prompt_idx": i,
                "attack": pair.get("attack", "unknown"),
                "condition": cond_label,
                "prompt": pair[cond_key],
                "core_text": pair.get("core_text", ""),
                "response": resp,
            })

    elapsed = time.perf_counter() - t0
    logger.info(f"Generation finished in {elapsed:.1f}s ({elapsed / 60:.1f} min)")

    # Save locally then copy to volume
    local_out = f"/workspace/data/processed/{output}"
    with open(local_out, "w") as f:
        for row in dump_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info(f"Wrote {len(dump_rows)} rows to {local_out}")

    os.makedirs(f"{DATA_DIR}/processed", exist_ok=True)
    shutil.copy(local_out, f"{DATA_DIR}/processed/{output}")
    data_volume.commit()
    logger.info(f"Saved to volume: {DATA_DIR}/processed/{output}")

    # Quick regex-based ASR summary
    from fact.evaluate import is_refusal
    clean_rows = [r for r in dump_rows if r["condition"] == "clean"]
    wrapped_rows = [r for r in dump_rows if r["condition"] == "wrapped"]
    base_refusal = sum(is_refusal(r["response"]) for r in clean_rows) / len(clean_rows)
    wrapped_asr = 1.0 - sum(is_refusal(r["response"]) for r in wrapped_rows) / len(wrapped_rows)
    print(f"\n=== REGEX SUMMARY ({len(pairs)} pairs) ===")
    print(f"Base refusal rate (clean):   {base_refusal:.2%}")
    print(f"Wrapped ASR (jailbroken):    {wrapped_asr:.2%}")


@app.function(
    image=workspace_image,
    gpu=GPU_CONFIG,
    volumes={
        MODEL_CACHE_DIR: model_volume,
        DATA_DIR: data_volume,
    },
    timeout=TIMEOUT,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
    ],
)
def smoke_jailbreak_responses(
    model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    n_samples: int = 20,
    max_new_tokens: int = 150,
):
    """Smoke test: generate responses for the first n_samples jailbreak pairs.

    Calls generate_jailbreak_responses with n_samples capped, output saved as
    jailbreak_responses_smoke.jsonl.  Use this to verify the pipeline before
    running the full dataset.

    Usage:
        modal run modal_app/run_experiment.py::smoke_jailbreak_responses
    """
    generate_jailbreak_responses.local(
        model=model,
        n_samples=n_samples,
        max_new_tokens=max_new_tokens,
        output="jailbreak_responses_smoke.jsonl",
    )


@app.function(
    image=workspace_image,
    gpu="A100",   # 40GB sufficient — ablation skips capability evals (no MMLU loglikelihood)
    volumes={
        MODEL_CACHE_DIR: model_volume,
        DATA_DIR: data_volume,
    },
    timeout=TIMEOUT,
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
    ],
)
def ablation_layer_eval(
    model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    n_samples: int = 200,
    epochs: int = 2,
    lr: float = 1e-4,
    fact_loss_weight: float = 0.2,
):
    """Layer-skip ablation: train 3 FACT adapters and evaluate each with bootstrap CIs.

    Trains three adapters on 400 samples (200 jailbreak + 200 sycophancy each):
      - all_layers:  no layer skipping (fact_skip_layers="")
      - skip6:       skip early 6 layers (fact_skip_layers="0 1 2 3 4 5")
      - skip12:      skip early 12 layers (fact_skip_layers="0 1 2 3 4 5 6 7 8 9 10 11")

    After training, runs behavioral evaluation (200 val samples, no capability evals)
    for each adapter. Saves per-sample response dumps (JSONL) for bootstrap CI computation.
    Run experiments/compute_ablation_ci.py locally to compute CIs.

    Usage:
        modal run --detach modal_app/run_experiment.py::ablation_layer_eval
    """
    import subprocess
    import sys
    import shutil
    import os
    import time

    sys.path.insert(0, "/workspace")

    ABLATIONS = [
        ("all_layers", ""),
        ("skip6",      "0 1 2 3 4 5"),
        ("skip12",     "0 1 2 3 4 5 6 7 8 9 10 11"),
    ]

    # Copy processed data from volume to workspace
    os.makedirs("/workspace/data/processed", exist_ok=True)
    for fname in ["jailbreak_all_responses.jsonl", "sycophancy_strong_cues_all_responses.jsonl", "if_dataset.jsonl"]:
        src = f"{DATA_DIR}/processed/{fname}"
        dst = f"/workspace/data/processed/{fname}"
        if Path(src).exists() and not Path(dst).exists():
            shutil.copy(src, dst)

    # -----------------------------------------------------------------------
    # Phase 1: Train all 3 adapters sequentially (single A100, share GPU)
    # -----------------------------------------------------------------------
    adapters = {}
    for name, skip_layers in ABLATIONS:
        adapter_output = f"adapters/ablation_lora_{name}"
        out_dir = f"/workspace/results/{adapter_output}"
        os.makedirs(out_dir, exist_ok=True)

        cmd = [
            sys.executable, "/workspace/experiments/run_fact.py",
            "--model", model,
            "--device", "cuda",
            "--n-samples", str(n_samples),   # per task; 200 jailbreak + 200 sycophancy = 400 total
            "--epochs", str(epochs),
            "--lr", str(lr),
            "--fact-loss-weight", str(fact_loss_weight),
            "--if-share", "0.3",
            "--adapter-output", out_dir,
        ]
        if skip_layers:
            cmd += ["--fact-skip-layers"] + skip_layers.split()

        print(f"\n{'='*60}")
        print(f"Training ablation: {name}  (skip_layers={skip_layers!r})")
        print(f"{'='*60}")
        print(f"Running: {' '.join(cmd)}")
        t0 = time.perf_counter()
        result = subprocess.run(cmd, cwd="/workspace")
        elapsed = time.perf_counter() - t0
        print(f"Training [{name}] finished in {elapsed:.1f}s ({elapsed/60:.1f} min)")

        if result.returncode != 0:
            raise RuntimeError(f"run_fact.py failed for ablation '{name}'")

        # Copy adapter to volume
        dest_dir = f"{DATA_DIR}/{adapter_output}"
        os.makedirs(dest_dir, exist_ok=True)
        for item in os.listdir(out_dir):
            src = os.path.join(out_dir, item)
            dst = os.path.join(dest_dir, item)
            if os.path.isfile(src):
                shutil.copy(src, dst)
            else:
                shutil.copytree(src, dst, dirs_exist_ok=True)
        data_volume.commit()
        print(f"Adapter [{name}] saved to {dest_dir}")
        adapters[name] = dest_dir

    # -----------------------------------------------------------------------
    # Phase 2: Behavioral evaluation — one run per adapter (full 200-sample val set)
    # Bootstrap CIs computed locally from per-sample JSONL dumps.
    # -----------------------------------------------------------------------
    jailbreak_arg  = "/workspace/data/processed/jailbreak_all_responses.jsonl"
    sycophancy_arg = "/workspace/data/processed/sycophancy_strong_cues_all_responses.jsonl"

    os.makedirs("/workspace/results/ablation", exist_ok=True)

    for name, _ in ABLATIONS:
        adapter_full = adapters[name]
        out_path  = f"/workspace/results/ablation/ablation_{name}.json"
        dump_path = f"/workspace/results/ablation/ablation_{name}_responses.jsonl"

        cmd = [
            sys.executable, "/workspace/experiments/run_behavioral_validation.py",
            "--model", model,
            "--adapter-path", adapter_full,
            "--n-samples", str(n_samples),
            "--max-new-tokens", "100",
            "--device", "cuda",
            "--jailbreak-data", jailbreak_arg,
            "--sycophancy-data", sycophancy_arg,
            "--output", out_path,
            "--dump", dump_path,
            "--capability-evals", "",   # skip cap evals — CI is on behavioral metrics only
        ]

        print(f"\n{'='*60}")
        print(f"Evaluating ablation: {name}")
        print(f"{'='*60}")
        print(f"Running: {' '.join(cmd)}")
        t0 = time.perf_counter()
        env = os.environ.copy()
        env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
        result = subprocess.run(cmd, cwd="/workspace", env=env)
        elapsed = time.perf_counter() - t0
        print(f"Eval [{name}] finished in {elapsed:.1f}s ({elapsed/60:.1f} min)")

        if result.returncode != 0:
            raise RuntimeError(f"run_behavioral_validation.py failed for ablation '{name}'")

        # Save eval outputs to volume
        os.makedirs(f"{DATA_DIR}/results/ablation", exist_ok=True)
        shutil.copy(out_path,  f"{DATA_DIR}/results/ablation/ablation_{name}.json")
        shutil.copy(dump_path, f"{DATA_DIR}/results/ablation/ablation_{name}_responses.jsonl")
        data_volume.commit()
        print(f"Results [{name}] saved to volume: results/ablation/ablation_{name}.*")

    print("\n" + "="*60)
    print("Ablation training + eval complete.")
    print("Download results:")
    print("  modal volume get fact-data results/ablation/ ./results/ablation/")
    print("Then compute bootstrap CIs:")
    print("  python experiments/compute_ablation_ci.py --results-dir results/ablation/")
    print("="*60)
