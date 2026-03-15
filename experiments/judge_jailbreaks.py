"""Judge jailbreak responses with the StrongREJECT LLM judge (runs locally).

Reads a JSONL dump produced by the Modal generate_jailbreak_responses function,
runs strongreject_judge on clean + wrapped rows, and reports:
  - Base ASR (clean harmful prompts — should be low, i.e. model mostly refuses)
  - Wrapped ASR (jailbreak-wrapped prompts — should be high if jailbreaks work)
  - Per-attack breakdown for both conditions

Results are saved as a JSON file.

Usage (requires OPENAI_API_KEY in env or .env file):

    # After downloading from Modal:
    modal volume get fact-data processed/jailbreak_responses.jsonl ./data/processed/
    python experiments/judge_jailbreaks.py \\
        --dump data/processed/jailbreak_responses.jsonl \\
        --output results/jailbreak_judge_results.json

    # Smoke test only (first 20 rows of each condition):
    python experiments/judge_jailbreaks.py \\
        --dump data/processed/jailbreak_responses_smoke.jsonl

Cost estimate (gpt-4o-mini, ~3088 rows): ~$0.047 total.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from fact.evaluate import is_refusal, strongreject_is_success, strongreject_judge
from fact.utils import get_logger

logger = get_logger("judge_jailbreaks")


def asr_stats(rows: list[dict], verdicts: list[dict], label: str, response_key: str) -> dict:
    n = len(verdicts)
    n_success_judge = sum(1 for v in verdicts if strongreject_is_success(v))
    n_success_regex = sum(1 for r in rows if not is_refusal(r[response_key]))
    scores = [v["score"] for v in verdicts if v.get("score") is not None]
    avg_score = sum(scores) / len(scores) if scores else None

    by_attack: dict[str, dict] = defaultdict(lambda: {"n": 0, "success_judge": 0, "success_regex": 0})
    for row, v in zip(rows, verdicts):
        atk = row.get("attack", "unknown")
        by_attack[atk]["n"] += 1
        if strongreject_is_success(v):
            by_attack[atk]["success_judge"] += 1
        if not is_refusal(row[response_key]):
            by_attack[atk]["success_regex"] += 1

    print(f"\n=== {label.upper()} ({n} items) ===")
    print(f"  ASR (judge) : {n_success_judge/n:.2%}  ({n_success_judge}/{n})")
    print(f"  ASR (regex) : {n_success_regex/n:.2%}  ({n_success_regex}/{n})")
    if avg_score is not None:
        print(f"  Avg score   : {avg_score:.2f} / 5.0")
    print("  Per-attack breakdown:")
    for atk in sorted(by_attack):
        d = by_attack[atk]
        print(f"    {atk:<12} n={d['n']:>4}  judge={d['success_judge']/d['n']:.0%}  regex={d['success_regex']/d['n']:.0%}")

    return {
        "n": n,
        "asr_judge": n_success_judge / n,
        "asr_regex": n_success_regex / n,
        "avg_score": avg_score,
        "by_attack": {
            atk: {
                "n": d["n"],
                "asr_judge": d["success_judge"] / d["n"],
                "asr_regex": d["success_regex"] / d["n"],
            }
            for atk, d in by_attack.items()
        },
    }


def main(args: argparse.Namespace) -> None:
    if not os.path.exists(args.dump):
        logger.error(f"Dump file not found: {args.dump}")
        logger.error("Run generate_jailbreak_responses on Modal first, then download with:")
        logger.error("  modal volume get fact-data processed/jailbreak_responses.jsonl ./data/processed/")
        sys.exit(1)

    rows = []
    with open(args.dump) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if args.n_samples > 0:
        rows = rows[:args.n_samples]
    logger.info(f"Loaded {len(rows)} rows from {args.dump}")

    logger.info(f"  {len(rows)} paired rows (clean + wrapped per row)")

    # Paired format: each row has clean_response / wrapped_response
    def _run_judge(responses: list[str], prompts: list[str], label: str) -> list[dict]:
        logger.info(f"Judging {label} ({len(responses)} items, max_workers={args.max_workers})…")
        return strongreject_judge(responses, prompts, model=args.model, max_workers=args.max_workers)

    core_prompts = [r.get("core_text") or r["clean"] for r in rows]

    # Load cached verdicts from a previous partial run if available
    cached: dict[str, list] = {"clean": None, "wrapped": None}
    if args.load_verdicts and Path(args.load_verdicts).exists():
        prev = [json.loads(l) for l in open(args.load_verdicts)]
        if len(prev) == len(rows):
            cached["clean"] = [p["clean_verdict"] for p in prev]
            cached["wrapped"] = [p["wrapped_verdict"] for p in prev]
            logger.info(f"Loaded cached verdicts from {args.load_verdicts}")

    run_clean = args.condition in ("both", "clean")
    run_wrapped = args.condition in ("both", "wrapped")

    clean_verdicts = (
        cached["clean"] if cached["clean"] and not run_clean
        else (_run_judge([r["clean_response"] for r in rows], core_prompts, "clean (base)") if run_clean else
              [{"score": None, "refusal": None, "convincing": None, "specific": None} for _ in rows])
    )
    wrapped_verdicts = (
        cached["wrapped"] if cached["wrapped"] and not run_wrapped
        else (_run_judge([r["wrapped_response"] for r in rows], core_prompts, "wrapped (jailbreak)") if run_wrapped else
              [{"score": None, "refusal": None, "convincing": None, "specific": None} for _ in rows])
    )

    clean_stats = asr_stats(rows, clean_verdicts, "base — clean harmful prompts", "clean_response")
    wrapped_stats = asr_stats(rows, wrapped_verdicts, "jailbreak — wrapped prompts", "wrapped_response")

    results = {
        "dump": args.dump,
        "judge_model": args.model,
        "base_clean": clean_stats,
        "jailbreak_wrapped": wrapped_stats,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {out_path}")

    # Save per-item verdicts to JSONL for reuse / inspection
    verdicts_path = out_path.with_stem(out_path.stem + "_verdicts").with_suffix(".jsonl")
    with open(verdicts_path, "w") as f:
        for row, cv, wv in zip(rows, clean_verdicts, wrapped_verdicts):
            entry = {
                "attack": row.get("attack"),
                "core_text": row.get("core_text", ""),
                "clean_response": row["clean_response"],
                "wrapped_response": row["wrapped_response"],
                "clean_refused_regex": is_refusal(row["clean_response"]),
                "wrapped_refused_regex": is_refusal(row["wrapped_response"]),
                "clean_verdict": cv,
                "wrapped_verdict": wv,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logger.info(f"Per-item verdicts saved to {verdicts_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dump", default="data/processed/jailbreak_all_responses.jsonl",
        help="Paired JSONL dump (default: data/processed/jailbreak_all_responses.jsonl)"
    )
    parser.add_argument("--n-samples", type=int, default=0, help="Cap number of rows to judge (0 = all)")
    parser.add_argument("--condition", default="both", choices=["both", "clean", "wrapped"],
                        help="Which condition to judge (default: both)")
    parser.add_argument("--load-verdicts", default=None,
                        help="Path to a _verdicts.jsonl from a previous run to reuse cached results")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI judge model (default gpt-4o-mini)")
    parser.add_argument("--max-workers", type=int, default=20, help="Parallel API calls (default 20)")
    parser.add_argument(
        "--output", default="results/jailbreak_judge_results.json",
        help="Output JSON path (default: results/jailbreak_judge_results.json)"
    )
    main(parser.parse_args())
