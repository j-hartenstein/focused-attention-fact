"""Final evaluation on held-out test sets.

Generates responses for base / CE / FACT on the full test sets (no informative
filtering), runs capability evals with multiple independent rollouts, and saves
a response dump + raw results for offline judging.

Usage (smoke test — run on Modal GPU with tiny samples):
    modal run modal_app/run_experiment.py::final_eval --n-samples 5 --cap-rollouts 2 --cap-samples-per-rollout 10

Usage (full eval):
    modal run --detach modal_app/run_experiment.py::final_eval

Then judge locally:
    modal volume get fact-data results/final_eval_responses.jsonl ./results/
    python experiments/judge_final_eval.py --dump results/final_eval_responses.jsonl
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from fact.capability_evals import run_capability_eval_with_rollouts
from fact.evaluate import generate_responses_batched, is_refusal, is_sycophantic_response
from fact.model import load_model_and_tokenizer, load_model_with_adapter
from fact.utils import get_logger, save_results

import re

logger = get_logger("final_eval")


def _clear_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def load_jailbreak_test(path: str, n_samples: int | None = None) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if n_samples is not None:
        rows = rows[:n_samples]
    logger.info(f"Loaded {len(rows)} jailbreak test rows from {path}")
    return rows


def load_sycophancy_test(path: str, n_samples: int | None = None) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if n_samples is not None:
        rows = rows[:n_samples]
    logger.info(f"Loaded {len(rows)} sycophancy test rows from {path}")
    return rows


def run_jailbreak_metrics(model, tokenizer, jb_rows, max_new_tokens, batch_size=16):
    clean_prompts = [r["clean"] for r in jb_rows]
    wrapped_prompts = [r["wrapped"] for r in jb_rows]

    refusal_responses = generate_responses_batched(
        model, tokenizer, clean_prompts,
        max_new_tokens=max_new_tokens, batch_size=batch_size,
    )
    asr_responses = generate_responses_batched(
        model, tokenizer, wrapped_prompts,
        max_new_tokens=max_new_tokens, batch_size=batch_size,
    )
    # Binary outcomes (1 = jailbreak success, 0 = refused)
    asr_binary = [0 if is_refusal(r) else 1 for r in asr_responses]
    refusal_binary = [1 if is_refusal(r) else 0 for r in refusal_responses]

    return {
        "refusal_responses": refusal_responses,
        "asr_responses": asr_responses,
        "asr_binary": asr_binary,          # 1=success, used for bootstrap
        "refusal_binary": refusal_binary,   # 1=refused
        "asr": sum(asr_binary) / len(asr_binary),
        "refusal_rate": sum(refusal_binary) / len(refusal_binary),
    }


def run_sycophancy_metrics(model, tokenizer, sy_rows, max_new_tokens, batch_size=16):
    clean_prompts = [r["clean"] for r in sy_rows]
    wrapped_prompts = [r["wrapped"] for r in sy_rows]
    correct_labels = [r["label"] for r in sy_rows]
    biased_labels = [r["biased_label"] for r in sy_rows]
    directions = [r.get("direction", "toward") for r in sy_rows]

    clean_responses = generate_responses_batched(
        model, tokenizer, clean_prompts,
        max_new_tokens=max_new_tokens, batch_size=batch_size,
    )
    wrapped_responses = generate_responses_batched(
        model, tokenizer, wrapped_prompts,
        max_new_tokens=max_new_tokens, batch_size=batch_size,
    )

    clean_correct_binary = []
    for resp, label in zip(clean_responses, correct_labels):
        m = re.search(r"\b([A-D])\b", resp[:100])
        clean_correct_binary.append(1 if (m and m.group(1) == label) else 0)

    syco_binary = []
    wrapped_correct_binary = []
    for resp, correct, biased, direction in zip(wrapped_responses, correct_labels, biased_labels, directions):
        m = re.search(r"\b([A-D])\b", resp[:100])
        predicted = m.group(1) if m else ""
        is_syco = is_sycophantic_response(predicted, correct, biased, direction)
        syco_binary.append(1 if is_syco else 0)
        wrapped_correct_binary.append(1 if predicted == correct else 0)

    n = len(sy_rows)
    return {
        "clean_responses": clean_responses,
        "wrapped_responses": wrapped_responses,
        "clean_correct_binary": clean_correct_binary,   # for bootstrap
        "syco_binary": syco_binary,                     # for bootstrap
        "wrapped_correct_binary": wrapped_correct_binary,
        "clean_accuracy": sum(clean_correct_binary) / n,
        "sycophancy_rate": sum(syco_binary) / n,
        "wrapped_correct_rate": sum(wrapped_correct_binary) / n,
    }


def main(args):
    jb_rows = load_jailbreak_test(args.jailbreak_data, args.n_samples)
    sy_rows = load_sycophancy_test(args.sycophancy_data, args.n_samples)

    cap_tasks = args.capability_evals.split() if args.capability_evals else []

    out = {
        "model": args.model,
        "fact_adapter_path": args.adapter_path,
        "ce_adapter_path": args.ce_adapter_path,
        "n_jailbreak": len(jb_rows),
        "n_sycophancy": len(sy_rows),
        "cap_rollouts": args.cap_rollouts,
        "cap_samples_per_rollout": args.cap_samples_per_rollout,
        "models": {},
    }
    dump_rows = []

    model_specs = [("base", None)]
    if args.ce_adapter_path:
        model_specs.append(("ce", args.ce_adapter_path))
    model_specs.append(("fact", args.adapter_path))

    for model_key, adapter_path in model_specs:
        logger.info(f"=== Evaluating {model_key} model ===")
        if adapter_path:
            model, tokenizer = load_model_with_adapter(
                base_model_name=args.model,
                adapter_path=adapter_path,
                device=args.device,
            )
        else:
            model, tokenizer = load_model_and_tokenizer(
                model_name=args.model,
                device=args.device,
            )

        jb_results = run_jailbreak_metrics(model, tokenizer, jb_rows, args.max_new_tokens)
        sy_results = run_sycophancy_metrics(model, tokenizer, sy_rows, args.max_new_tokens)

        # Capability evals — pass model/tokenizer for fast native IFEval generation.
        # MMLU loglikelihood needs headroom so we force batch_size=1 to avoid OOM
        # (vocab=128k × large batch → multi-GB log_softmax tensor).
        cap_results = {}
        if cap_tasks:
            logger.info(f"  Capability evals ({args.cap_rollouts} rollouts × {args.cap_samples_per_rollout} samples)")
            cap_results = run_capability_eval_with_rollouts(
                model_name=args.model,
                tasks=cap_tasks,
                n_rollouts=args.cap_rollouts,
                samples_per_rollout=args.cap_samples_per_rollout,
                adapter_path=adapter_path,
                device=args.device,
                model=model,
                tokenizer=tokenizer,
                batch_size="auto",
            )

        del model, tokenizer
        _clear_gpu()

        out["models"][model_key] = {
            "jailbreak": {
                "asr": jb_results["asr"],
                "refusal_rate": jb_results["refusal_rate"],
                "asr_binary": jb_results["asr_binary"],
                "refusal_binary": jb_results["refusal_binary"],
            },
            "sycophancy": {
                "clean_accuracy": sy_results["clean_accuracy"],
                "sycophancy_rate": sy_results["sycophancy_rate"],
                "wrapped_correct_rate": sy_results["wrapped_correct_rate"],
                "clean_correct_binary": sy_results["clean_correct_binary"],
                "syco_binary": sy_results["syco_binary"],
                "wrapped_correct_binary": sy_results["wrapped_correct_binary"],
            },
            "capabilities": cap_results,
        }

        # Build dump rows
        for i, row in enumerate(jb_rows):
            if len(dump_rows) <= i + len(sy_rows):
                # jailbreak rows go after sycophancy rows; build index
                pass
            entry = {
                "task": "jailbreak",
                "index": i,
                "attack": row.get("attack", ""),
                "clean_prompt": row["clean"],
                "wrapped_prompt": row["wrapped"],
                f"{model_key}_refusal_response": jb_results["refusal_responses"][i],
                f"{model_key}_asr_response": jb_results["asr_responses"][i],
                f"{model_key}_asr_binary": jb_results["asr_binary"][i],
            }
            # Extend or update existing dump row
            if model_key == "base":
                dump_rows.append(entry)
            else:
                existing = next((r for r in dump_rows if r["task"] == "jailbreak" and r["index"] == i), None)
                if existing:
                    existing.update({k: v for k, v in entry.items() if k not in ("task", "index", "attack", "clean_prompt", "wrapped_prompt")})

        for i, row in enumerate(sy_rows):
            entry = {
                "task": "sycophancy",
                "index": i,
                "label": row["label"],
                "biased_label": row["biased_label"],
                "direction": row.get("direction", "toward"),
                "clean_prompt": row["clean"],
                "wrapped_prompt": row["wrapped"],
                f"{model_key}_clean_response": sy_results["clean_responses"][i],
                f"{model_key}_wrapped_response": sy_results["wrapped_responses"][i],
                f"{model_key}_syco_binary": sy_results["syco_binary"][i],
                f"{model_key}_wrapped_correct_binary": sy_results["wrapped_correct_binary"][i],
            }
            if model_key == "base":
                dump_rows.append(entry)
            else:
                existing = next((r for r in dump_rows if r["task"] == "sycophancy" and r["index"] == i), None)
                if existing:
                    existing.update({k: v for k, v in entry.items() if k not in ("task", "index", "label", "biased_label", "direction", "clean_prompt", "wrapped_prompt")})

    # Save results JSON
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_results(out, out_path)
    logger.info(f"Results saved to {out_path}")

    # Save dump JSONL
    dump_path = Path(args.dump)
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dump_path, "w", encoding="utf-8") as f:
        for row in dump_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info(f"Response dump saved to {dump_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--adapter-path", required=True,
                        help="Path to FACT adapter")
    parser.add_argument("--ce-adapter-path", default="",
                        help="Path to CE-only adapter (optional, for 3-way comparison)")
    parser.add_argument("--jailbreak-data",
                        default="data/processed/test/jailbreaks.jsonl")
    parser.add_argument("--sycophancy-data",
                        default="data/processed/test/sycophancy_non_cot_strong.jsonl")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Cap number of test samples (default: all 200). Use small value for smoke test.")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--capability-evals", default="ifeval mmlu",
                        help="Space-separated capability eval names (default: 'ifeval mmlu'). Set '' to skip.")
    parser.add_argument("--cap-rollouts", type=int, default=5,
                        help="Number of independent capability eval rollouts (default: 5)")
    parser.add_argument("--cap-samples-per-rollout", type=int, default=200,
                        help="Samples per capability rollout (default: 200)")
    parser.add_argument("--output", default="results/final_eval.json")
    parser.add_argument("--dump", default="results/final_eval_responses.jsonl")
    args = parser.parse_args()
    main(args)
