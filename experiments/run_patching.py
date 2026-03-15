"""Phase 1: Attention patching experiment.

Sweeps over 4 coarse patching conditions (all / early / mid / late layers) for
jailbreak and sycophancy tasks. Reads pre-filtered informative pairs directly from
data/patching/ (no baseline generation needed to select pairs; all pairs already
satisfy clean-correct + wrapped-incorrect).

Usage:
    python experiments/run_patching.py --n-samples 200 --device cuda
    python experiments/run_patching.py --n-samples 20 --device cpu  # local smoke test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from fact.data import (
    PATCHING_DATA,
    PATCHING_LOADERS,
    PromptPairDataset,
)
from fact.model import load_model_and_tokenizer, num_layers, layer_subsets
from fact.patching import run_patching_experiment, summarize_results, PatchingResult
from fact.utils import get_logger, save_results

logger = get_logger("patching")


def main(args):
    t_start = time.perf_counter()

    model, tokenizer = load_model_and_tokenizer(
        model_name=args.model,
        device=args.device,
    )
    n = num_layers(model)
    logger.info(f"Model has {n} layers. Patching conditions: {layer_subsets(n)}")

    all_results: list[PatchingResult] = []
    timing: dict[str, float] = {}

    tasks = args.tasks if args.tasks else ["jailbreak", "sycophancy"]

    for task in tasks:
        logger.info(f"\n=== Task: {task} ===")
        data_path = PATCHING_DATA[task]
        kwargs = {"n_samples": args.n_samples}
        if task == "jailbreak" and args.attacks:
            kwargs["attacks"] = args.attacks
        pairs = PATCHING_LOADERS[task](data_path, tokenizer, **kwargs)
        dataset = PromptPairDataset(pairs, tokenizer, max_length=args.max_length)

        logger.info(f"Running patching on {len(dataset)} pairs...")
        t_task = time.perf_counter()
        task_results = run_patching_experiment(
            model,
            tokenizer,
            dataset.items,
            max_new_tokens=args.max_new_tokens,
            conditions=args.conditions if args.conditions else None,
            batch_size=args.batch_size,
            last_position_only=args.last_position_only,
            last_position_noncore_mode=args.last_position_noncore_mode,
            last_position_noncore_epsilon=args.last_position_noncore_epsilon,
            last_position_noncore_beta=args.last_position_noncore_beta,
            full_matrix_noncore_mode=args.full_matrix_noncore_mode,
            full_matrix_noncore_beta=args.full_matrix_noncore_beta,
        )
        task_elapsed = time.perf_counter() - t_task
        timing[task] = task_elapsed
        logger.info(f"Task {task} finished in {task_elapsed:.1f}s "
                    f"({task_elapsed/max(len(task_results),1):.1f}s/pair)")
        all_results.extend(task_results)

    total_elapsed = time.perf_counter() - t_start

    # Summarize
    summary = summarize_results(all_results)
    logger.info("\n=== PATCHING SUMMARY ===")
    for task, by_cond in summary["by_task"].items():
        logger.info(f"\nTask: {task}")
        for cond, stats in by_cond.items():
            logger.info(f"  {cond:8s}: recovery_rate={stats['recovery_rate']:.2%}  "
                        f"({stats['n_recovered']}/{stats['n']})")

    # Serialize results (exclude pair objects for JSON serialization)
    serializable = []
    for r in all_results:
        serializable.append({
            "task": r.pair.task,
            "source": r.pair.source,
            "attack": r.pair.attack,
            "label": r.pair.label,
            "biased_label": r.pair.biased_label,
            "direction": r.pair.direction,
            "n_core_tokens": len(r.overlap_clean_indices),
            "clean_prompt": r.pair.clean,
            "wrapped_prompt": r.pair.wrapped,
            "clean_correct": r.clean_correct,
            "wrapped_correct": r.wrapped_correct,
            "baseline_clean_response": r.baseline_clean_response,
            "baseline_wrapped_response": r.baseline_wrapped_response,
            "patched_responses": r.patched_responses,
            "patch_recovers": r.patch_recovers,
        })

    out_path = Path(args.output)
    timing["total"] = total_elapsed
    save_results({"summary": summary, "timing": timing, "results": serializable}, out_path)
    logger.info(f"\nResults saved to {out_path}")

    # Print quick table
    print("\n=== RECOVERY RATES ===")
    print(f"{'task':<14} {'condition':<10} {'rate':>8}  n_informative")
    print("-" * 50)
    for task, by_cond in summary["by_task"].items():
        for cond, stats in by_cond.items():
            print(f"{task:<14} {cond:<10} {stats['recovery_rate']:>7.1%}  {stats['n']}")

    print("\n=== TIMING ===")
    for task in tasks:
        if task in timing:
            n_pairs = sum(1 for r in all_results if r.pair.task == task)
            print(f"  {task:<14}: {timing[task]:.1f}s  ({timing[task]/max(n_pairs,1):.1f}s/pair, {n_pairs} pairs)")
    print(f"  {'total':<14}: {total_elapsed:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Pairs to sample per task (default: all pairs in each dataset)")
    parser.add_argument("--max-new-tokens", type=int, default=70)
    parser.add_argument("--max-length", type=int, default=2000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tasks", nargs="+", choices=["jailbreak", "sycophancy"],
                        default=None, help="Tasks to run (default: both)")
    parser.add_argument("--attacks", nargs="+",
                        choices=["autodan", "msj", "prefill", "human_mt", "gcg", "best_of_n"],
                        default=None, help="Jailbreak attack types to include (default: all)")
    parser.add_argument("--conditions", nargs="+", choices=["all", "early", "mid", "late"],
                        default=None, help="Patching conditions to run (default: all 4)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for baseline clean/wrapped generation")
    parser.add_argument("--last-position-only", action="store_true",
                        help="Only patch the last prefill token (assistant/generation trigger) "
                             "instead of the full attention matrix. More targeted: only the "
                             "model's generation decision is influenced, not all intermediate "
                             "representations. Patched row is renormalized to sum to 1.")
    parser.add_argument("--last-position-noncore-mode",
                        choices=["zero", "keep", "epsilon", "beta"],
                        default="zero",
                        help="When --last-position-only is enabled, choose how to fill "
                             "non-core keys in the patched last row before renormalization.")
    parser.add_argument("--last-position-noncore-epsilon", type=float, default=1e-5,
                        help="Epsilon value used for non-core keys when "
                             "--last-position-noncore-mode=epsilon.")
    parser.add_argument("--last-position-noncore-beta", type=float, default=0.01,
                        help="Target non-core mass used when "
                             "--last-position-noncore-mode=beta.")
    parser.add_argument("--full-matrix-noncore-mode",
                        choices=["zero", "beta"],
                        default="zero",
                        help="When NOT using --last-position-only, choose how to fill "
                             "non-core entries that are otherwise zeroed.")
    parser.add_argument("--full-matrix-noncore-beta", type=float, default=0.01,
                        help="Target non-core mass used when "
                             "--full-matrix-noncore-mode=beta.")
    parser.add_argument("--output", default="results/patching.json")
    args = parser.parse_args()
    main(args)
