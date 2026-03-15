"""Phase 2: FACT fine-tuning with LoRA.

Uses the same dataset and structure as the patching experiment (pre-filtered
informative pairs). Minimizes L_Inv on the last-row attention over I_core.

Usage:
    python experiments/run_fact.py --n-samples 20 --device cpu   # local smoke test
    python experiments/run_fact.py --n-samples 200 --device cuda # full run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from fact.data import FACT_DATA, IF_DATA, PATCHING_LOADERS, PromptPairDataset, load_if_pairs
from fact.model import load_model_and_tokenizer
from fact.training import collate_tokenized_pairs, get_lora_model, train_fact
from fact.utils import get_logger

logger = get_logger("fact")


def main(args) -> None:
    effective_device = args.device
    if args.device == "cuda" and not torch.cuda.is_available():
        effective_device = "cpu"
        logger.warning("CUDA not available, using CPU")
    elif args.device == "mps":
        if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            effective_device = "cpu"
            logger.warning("MPS not available, using CPU")

    model, tokenizer = load_model_and_tokenizer(
        model_name=args.model,
        device=effective_device,
    )
    model.train()

    # Same dataset as patching: pre-filtered informative pairs
    tasks = args.tasks or ["jailbreak", "sycophancy"]
    all_pairs = []
    for task in tasks:
        data_path = FACT_DATA[task]
        kwargs = {"n_samples": args.n_samples} if args.n_samples else {}
        kwargs["informative_only"] = True
        kwargs["split"] = args.split
        kwargs["val_size"] = args.val_size
        if task == "jailbreak" and args.attacks:
            kwargs["attacks"] = args.attacks
        pairs = PATCHING_LOADERS[task](data_path, tokenizer, **kwargs)
        all_pairs.extend(pairs)

    if not all_pairs:
        raise RuntimeError("No pairs loaded. Check PATCHING_DATA paths and --n-samples.")

    # --- Instruction-following (CE-only) samples ---
    if args.if_share > 0:
        n_adversarial = len(all_pairs)
        n_if = round(n_adversarial * args.if_share / (1 - args.if_share))
        if_pairs = load_if_pairs(IF_DATA, n_samples=n_if, seed=args.seed)
        logger.info(
            f"IF samples: {len(if_pairs)} (target {n_if}) → "
            f"total {n_adversarial + len(if_pairs)} "
            f"({100 * len(if_pairs) / (n_adversarial + len(if_pairs)):.1f}% IF)"
        )
        all_pairs = all_pairs + if_pairs

    dataset = PromptPairDataset(all_pairs, tokenizer, max_length=args.max_length)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    collate_fn = lambda batch: collate_tokenized_pairs(batch, pad_token_id=pad_id)
    train_loader = DataLoader(
        dataset.items,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    model = get_lora_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
    )
    # Single device for batch placement (when device_map=auto, use model's device)
    device = torch.device(effective_device) if args.device != "auto" else next(model.parameters()).device

    # Parse layer selection
    fact_layers = None
    if args.fact_skip_layers is not None:
        n_layers = model.config.num_hidden_layers
        fact_layers = [l for l in range(n_layers) if l not in args.fact_skip_layers]
        logger.info(f"FACT loss on {len(fact_layers)}/{n_layers} layers (skipping {sorted(args.fact_skip_layers)})")

    train_fact(
        model,
        tokenizer,
        train_loader,
        epochs=args.epochs,
        lr=args.lr,
        adapter_output_path=args.adapter_output,
        device=device,
        log_interval=args.log_interval,
        zero_clean_noncore=args.zero_clean_noncore,
        renormalize_clean_masked=args.renormalize_clean_masked,
        fact_loss_weight=args.fact_loss_weight,
        max_grad_norm=args.max_grad_norm,
        fact_layers=fact_layers,
        layer_eval_batches=args.layer_eval_batches,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FACT fine-tuning with LoRA")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct",
                        help="Base model (use 3B for local smoke test, 8B on Modal)")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu", "mps", "auto"),
                        help="Device: cuda, cpu, mps (Apple Silicon), or auto")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Max samples per task (default: all)")
    parser.add_argument("--tasks", nargs="+", choices=["jailbreak", "sycophancy"],
                        default=None, help="Tasks (default: both)")
    parser.add_argument("--attacks", nargs="+",
                        choices=["autodan", "msj", "prefill", "human_mt", "gcg", "best_of_n"],
                        default=None, help="Jailbreak attack filter (default: all)")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", nargs="+", default=None,
                        help="LoRA target modules (default: q_proj v_proj)")
    parser.add_argument("--adapter-output", default="results/adapters/fact_lora")
    parser.add_argument("--log-interval", type=int, default=10)
    # Loss variant: zero clean non-core and optionally renormalize
    parser.add_argument("--zero-clean-noncore", action="store_true",
                        help="Zero clean last-row attention at non-I_core keys before core loss")
    parser.add_argument("--renormalize-clean-masked", action="store_true",
                        help="Renormalize clean row after zeroing non-core (use with --zero-clean-noncore)")
    parser.add_argument("--fact-loss-weight", type=float, default=0.1,
                        help="Weight λ for FACT loss: total_loss = lm_loss + λ * fact_loss (default 0.1; try 0.05 for gentler)")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                        help="Max gradient norm for clipping (default 1.0)")
    parser.add_argument("--fact-skip-layers", type=int, nargs="+", default=None,
                        help="Layer indices to EXCLUDE from FACT loss (e.g. 0 1 2 3 4 5 to skip early layers)")
    parser.add_argument("--if-share", type=float, default=0.3,
                        help="Fraction of total samples that are CE-only IF samples (default: 0.3). Set 0 to disable.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for IF sample selection (default: 42)")
    parser.add_argument("--layer-eval-batches", type=int, default=0,
                        help="If >0, run per-layer L_Inv eval over this many batches at end of each epoch (default: 0 = disabled)")
    parser.add_argument("--split", default="train", choices=["train", "val"],
                        help="Which split to load: 'train' (default) or 'val'.")
    parser.add_argument("--val-size", type=int, default=200,
                        help="Number of informative pairs per task to hold out as val (default: 200).")
    parsed = parser.parse_args()
    main(parsed)
