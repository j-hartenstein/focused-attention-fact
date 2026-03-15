"""Evaluate attention invariance: compare mean L_Inv and per-layer L_Inv for base vs base+FACT.

Loads and evaluates the base model first, then base+adapter on the same pairs,
computes L_Inv (last-row, I_core) for both, and plots mean + per-layer comparison.
Lower L_Inv with FACT indicates more invariant attention.

Usage:
    python experiments/eval_attention_invariance.py --adapter-path results/adapters/fact_lora --n-samples 20 --device cpu
    python experiments/eval_attention_invariance.py --adapter-path results/adapters/fact_lora --n-samples 50 --device mps --output results/l_inv_plot.png
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from fact.data import PATCHING_DATA, PATCHING_LOADERS, PromptPairDataset
from fact.model import load_model_and_tokenizer, load_model_with_adapter
from fact.training import collate_tokenized_pairs, compute_fact_loss_breakdown
from fact.utils import get_logger

logger = get_logger("eval_l_inv")

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    plt = None
    np = None


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        return "cpu"
    if requested == "mps":
        if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            logger.warning("MPS not available, using CPU")
            return "cpu"
    return requested


def _cleanup_device_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _evaluate_model_l_inv(
    model,
    loader: DataLoader,
    device: torch.device,
    n_layers: int,
) -> tuple[float, list[float]]:
    mean_loss_list: list[float] = []
    per_layer_sum: list[float] = [0.0] * n_layers
    per_layer_count: list[int] = [0] * n_layers

    with torch.no_grad():
        for batch in loader:
            clean_ids = batch["clean_ids"].to(device)
            wrapped_ids = batch["wrapped_ids"].to(device)
            attention_mask_clean = batch["attention_mask_clean"].to(device)
            attention_mask_wrapped = batch["attention_mask_wrapped"].to(device)
            clean_core = batch["clean_core_indices"]
            wrapped_core = batch["wrapped_core_indices"]

            try:
                out_clean = model(
                    input_ids=clean_ids,
                    attention_mask=attention_mask_clean,
                    output_attentions=True,
                )
                out_wrapped = model(
                    input_ids=wrapped_ids,
                    attention_mask=attention_mask_wrapped,
                    output_attentions=True,
                )
            except torch.OutOfMemoryError as exc:
                _cleanup_device_memory(device)
                raise RuntimeError(
                    "CUDA OOM during attention invariance evaluation. "
                    "Try --batch-size 1 and/or a smaller --max-length "
                    "(e.g., 768 or 512)."
                ) from exc

            if out_clean.attentions is None or out_wrapped.attentions is None:
                raise RuntimeError("Model did not return attentions.")

            mean_loss, per_layer_loss, _ = compute_fact_loss_breakdown(
                out_clean.attentions,
                out_wrapped.attentions,
                clean_core,
                wrapped_core,
            )
            mean_loss_list.append(mean_loss)
            for layer_idx in range(n_layers):
                per_layer_sum[layer_idx] += per_layer_loss[layer_idx]
                per_layer_count[layer_idx] += 1

            del out_clean, out_wrapped
            del clean_ids, wrapped_ids, attention_mask_clean, attention_mask_wrapped
            if device.type == "cuda":
                torch.cuda.empty_cache()

    mean_loss = sum(mean_loss_list) / len(mean_loss_list) if mean_loss_list else 0.0
    per_layer_means = [
        per_layer_sum[layer_idx] / per_layer_count[layer_idx]
        if per_layer_count[layer_idx]
        else 0.0
        for layer_idx in range(n_layers)
    ]
    return mean_loss, per_layer_means


def main(args) -> None:
    effective_device = _resolve_device(args.device)

    device = torch.device(effective_device)

    # Load base model first and keep tokenizer for shared dataset tokenization.
    logger.info("Loading base model...")
    base_model, tokenizer = load_model_and_tokenizer(
        model_name=args.base_model,
        device=effective_device,
    )
    base_model.eval()
    n_layers = len(base_model.model.layers)

    # Same data as patching/FACT training
    tasks = args.tasks or ["jailbreak", "sycophancy"]
    all_pairs = []
    for task in tasks:
        data_path = PATCHING_DATA[task]
        kwargs = {"n_samples": args.n_samples} if args.n_samples else {}
        if task == "jailbreak" and args.attacks:
            kwargs["attacks"] = args.attacks
        pairs = PATCHING_LOADERS[task](data_path, tokenizer, **kwargs)
        all_pairs.extend(pairs)

    if not all_pairs:
        raise RuntimeError("No pairs loaded. Check PATCHING_DATA paths and --n-samples.")

    dataset = PromptPairDataset(all_pairs, tokenizer, max_length=args.max_length)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    collate_fn = lambda batch: collate_tokenized_pairs(batch, pad_token_id=pad_id)
    loader = DataLoader(
        dataset.items,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Evaluate base model, then free GPU memory before loading base+adapter.
    logger.info("Evaluating base model...")
    mean_base, per_layer_base = _evaluate_model_l_inv(
        base_model,
        loader,
        device,
        n_layers,
    )
    del base_model
    _cleanup_device_memory(device)

    logger.info("Loading base + FACT adapter...")
    fact_model, _ = load_model_with_adapter(
        base_model_name=args.base_model,
        adapter_path=args.adapter_path,
        device=effective_device,
    )
    fact_model.eval()

    logger.info("Evaluating FACT model...")
    mean_fact, per_layer_fact = _evaluate_model_l_inv(
        fact_model,
        loader,
        device,
        n_layers,
    )
    del fact_model
    _cleanup_device_memory(device)

    logger.info(f"Mean L_Inv  base: {mean_base:.6f}")
    logger.info(f"Mean L_Inv  FACT: {mean_fact:.6f}  (lower = more invariant)")

    if plt is None or np is None:
        logger.warning("matplotlib/numpy not available; skipping plot. Install with: pip install matplotlib numpy")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Bar: mean base vs FACT
    ax = axes[0]
    ax.bar(["Base", "FACT"], [mean_base, mean_fact], color=["#1f77b4", "#2ca02c"])
    ax.set_ylabel("Mean L_Inv")
    ax.set_title("Attention invariance (lower = better)")
    ax.set_ylim(0, max(mean_base, mean_fact) * 1.15)

    # Lines: per-layer
    ax = axes[1]
    layers = list(range(n_layers))
    ax.plot(layers, per_layer_base, "o-", label="Base", color="#1f77b4")
    ax.plot(layers, per_layer_fact, "s-", label="FACT", color="#2ca02c")
    ax.set_xlabel("Layer")
    ax.set_ylabel("L_Inv (mean over pairs)")
    ax.set_title("L_Inv by layer")
    ax.legend()

    plt.tight_layout()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info(f"Plot saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate L_Inv: base vs base+FACT")
    parser.add_argument("--base-model", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--adapter-path", required=True, help="Path to saved FACT adapter (e.g. results/adapters/fact_lora)")
    parser.add_argument("--n-samples", type=int, default=None, help="Max pairs per task (default: all)")
    parser.add_argument("--tasks", nargs="+", choices=["jailbreak", "sycophancy"], default=None)
    parser.add_argument("--attacks", nargs="+", default=None)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu", "mps", "auto"))
    parser.add_argument("--output", default="results/l_inv_eval.png")
    args = parser.parse_args()
    main(args)
