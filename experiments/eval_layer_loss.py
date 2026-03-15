"""Compute per-layer L_Inv for one or more trained adapters (post-hoc, no retraining).

Loads each adapter, runs compute_fact_loss_breakdown over --n-batches of val pairs,
and saves a layer_history.json in the same format produced by train_fact so that
plot_layer_loss.py can consume it directly.

Usage:
    python experiments/eval_layer_loss.py \\
        --adapter-path /data/adapters/fact_lora_3ep_1e4lr_flw025 \\
        --output results/layer_history_skiplayers.json

    # Multiple adapters in one shot:
    python experiments/eval_layer_loss.py \\
        --adapter-path /data/adapters/fact_lora_3ep_1e4lr_flw025 \\
                       /data/adapters/fact_lora_3ep_1e4lr_flw025_alllayers \\
        --output results/layer_history_skiplayers.json \\
                 results/layer_history_alllayers.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from fact.data import FACT_DATA, PATCHING_LOADERS, PromptPairDataset
from fact.model import load_model_with_adapter
from fact.training import collate_tokenized_pairs, compute_fact_loss_breakdown
from fact.utils import get_logger

logger = get_logger("eval_layer_loss")


def eval_per_layer(model, dataloader, device, n_batches: int) -> list[float]:
    """Run compute_fact_loss_breakdown over n_batches, return per-layer means."""
    model.eval()
    n_layers = model.config.num_hidden_layers
    layer_sums = [0.0] * n_layers
    layer_counts = [0] * n_layers
    n_done = 0

    for batch in dataloader:
        if n_done >= n_batches:
            break
        fact_indices = batch["apply_fact"].nonzero(as_tuple=True)[0]
        if len(fact_indices) == 0:
            continue

        clean_ids  = batch["clean_ids"][fact_indices].to(device)
        wrapped_ids = batch["wrapped_ids"][fact_indices].to(device)
        mask_clean   = batch["attention_mask_clean"][fact_indices].to(device)
        mask_wrapped = batch["attention_mask_wrapped"][fact_indices].to(device)
        ci = [batch["clean_core_indices"][i] for i in fact_indices.tolist()]
        wi = [batch["wrapped_core_indices"][i] for i in fact_indices.tolist()]

        with torch.no_grad():
            attn_clean   = model(input_ids=clean_ids,   attention_mask=mask_clean,
                                 output_attentions=True).attentions
            attn_wrapped = model(input_ids=wrapped_ids, attention_mask=mask_wrapped,
                                 output_attentions=True).attentions

        _, per_layer, _ = compute_fact_loss_breakdown(attn_clean, attn_wrapped, ci, wi)
        for l, v in enumerate(per_layer):
            if v > 0:
                layer_sums[l] += v
                layer_counts[l] += 1
        n_done += 1
        if n_done % 10 == 0:
            logger.info(f"  {n_done}/{n_batches} batches done")

    return [
        layer_sums[l] / layer_counts[l] if layer_counts[l] > 0 else 0.0
        for l in range(n_layers)
    ]


def main(args: argparse.Namespace) -> None:
    if len(args.adapter_path) != len(args.output):
        raise ValueError("--adapter-path and --output must have the same number of entries")

    device = torch.device(args.device)

    # Build val dataloader once (shared across adapters — only data, no model)
    logger.info("Loading val data…")
    # Import tokenizer from first adapter to tokenize
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path[0])

    pairs = []
    for task in ["jailbreak", "sycophancy"]:
        task_pairs = PATCHING_LOADERS[task](
            FACT_DATA[task], tokenizer,
            informative_only=True, split="val", val_size=200,
        )
        pairs.extend(task_pairs)
    logger.info(f"Val pairs: {len(pairs)}")

    dataset = PromptPairDataset(pairs, tokenizer, max_length=1024)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    dataloader = DataLoader(
        dataset.items,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_tokenized_pairs(b, pad_token_id=pad_id),
    )

    for adapter_path, out_path in zip(args.adapter_path, args.output):
        logger.info(f"Loading adapter: {adapter_path}")
        model, _ = load_model_with_adapter(args.base_model, adapter_path, device=args.device)
        model.to(device)

        logger.info(f"Running per-layer eval ({args.n_batches} batches)…")
        per_layer = eval_per_layer(model, dataloader, device, args.n_batches)

        result = [{"epoch": "posthoc", "per_layer_means": per_layer}]
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"Saved to {out_path}")
        logger.info(f"  min={min(per_layer):.4f}  max={max(per_layer):.4f}  "
                    f"mean={sum(per_layer)/len(per_layer):.4f}")

        # Free GPU memory before next adapter
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter-path", nargs="+", required=True,
                        help="Path(s) to adapter directory(s)")
    parser.add_argument("--output", nargs="+", required=True,
                        help="Output JSON path(s) for layer_history (one per adapter)")
    parser.add_argument("--base-model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--n-batches", type=int, default=50,
                        help="Number of batches to eval over (default 50)")
    main(parser.parse_args())
