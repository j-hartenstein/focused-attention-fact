"""Download and prepare static validation sets for IFEval and MMLU.

Creates deterministic 100-sample JSONL files used as the static val set in
behavioral_validation. Run this once locally; the output files are committed
to the repo under data/capability_evals/.

Usage:
    python data/scripts/download_capability_evals.py
    python data/scripts/download_capability_evals.py --n-ifeval 100 --n-mmlu 100 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def download_ifeval(out_path: Path, n: int, seed: int) -> None:
    """Download google/IFEval and save a shuffled n-sample JSONL val set.

    IFEval has a single 'train' split (541 prompts). We shuffle and take n.
    Each row keeps only the fields needed by the lm_eval task:
        key, prompt, instruction_id_list, kwargs
    """
    from datasets import load_dataset

    print(f"Downloading google/IFEval ...")
    ds = load_dataset("google/IFEval", split="train")
    items = list(ds)
    rng = random.Random(seed)
    rng.shuffle(items)
    items = items[:n]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for item in items:
            row = {
                "key": item["key"],
                "prompt": item["prompt"],
                "instruction_id_list": item["instruction_id_list"],
                "kwargs": item["kwargs"],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  Saved {len(items)} IFEval prompts → {out_path}")


def download_mmlu(out_path: Path, n: int, seed: int) -> None:
    """Download cais/mmlu (validation split across all subjects) and save n-sample JSONL.

    Uses the 'all' config, validation split (1540 questions). We shuffle and take n.
    Each row keeps: question, subject, choices, answer (integer 0-3).
    """
    from datasets import load_dataset

    print(f"Downloading cais/mmlu (all subjects, validation split) ...")
    ds = load_dataset("cais/mmlu", "all", split="validation")
    items = list(ds)
    rng = random.Random(seed)
    rng.shuffle(items)
    items = items[:n]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for item in items:
            row = {
                "question": item["question"],
                "subject": item["subject"],
                "choices": list(item["choices"]),
                "answer": int(item["answer"]),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  Saved {len(items)} MMLU questions → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download IFEval + MMLU val sets")
    parser.add_argument("--n-ifeval", type=int, default=100, help="IFEval val set size (default: 100)")
    parser.add_argument("--n-mmlu", type=int, default=100, help="MMLU val set size (default: 100)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling (default: 42)")
    parser.add_argument("--out-dir", default="data/capability_evals",
                        help="Output directory (default: data/capability_evals)")
    args = parser.parse_args()

    root = Path(__file__).parent.parent.parent
    out_dir = root / args.out_dir

    download_ifeval(out_dir / "ifeval_val.jsonl", n=args.n_ifeval, seed=args.seed)
    download_mmlu(out_dir / "mmlu_val.jsonl", n=args.n_mmlu, seed=args.seed)

    print(f"\nDone. Val sets saved to {out_dir}/")
    print("Commit these files to the repo so they are available on Modal.")


if __name__ == "__main__":
    main()
