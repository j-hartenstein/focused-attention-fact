"""Compute bootstrap confidence intervals and plot for ablation results.

Reads per-sample JSONL dumps from the layer-skip ablation study and computes
95% bootstrap CIs for all behavioral metrics across 3 adapter configs.

Metrics plotted:
  - Jailbreak: clean refusal rate, wrapped ASR
  - Sycophancy: clean accuracy, wrapped sycophancy rate, wrapped correct rate

Usage:
    modal volume get fact-data results/ablation/ ./results/ablation/

    python experiments/compute_ablation_ci.py --results-dir results/ablation_v2/
    python experiments/compute_ablation_ci.py --results-dir results/ablation_v2/ --n-bootstrap 10000
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from fact.evaluate import is_refusal, is_sycophantic_response

ABLATIONS = ["all_layers", "skip6", "skip12"]
ABLATION_LABELS = {
    "all_layers": "All layers",
    "skip6":      "Skip first 6",
    "skip12":     "Skip first 12",
}


def bootstrap_ci(outcomes: list[bool], n_bootstrap: int = 5000, alpha: float = 0.05, seed: int = 42) -> tuple[float, float, float]:
    """Return (mean, ci_lower, ci_upper) via percentile bootstrap."""
    arr = np.array(outcomes, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(arr)
    means = np.array([rng.choice(arr, size=n, replace=True).mean() for _ in range(n_bootstrap)])
    lo = np.percentile(means, 100 * alpha / 2)
    hi = np.percentile(means, 100 * (1 - alpha / 2))
    return float(arr.mean()), float(lo), float(hi)


def ci_from_scalar(mean: float, n: int, n_bootstrap: int = 5000, seed: int = 42) -> tuple[float, float, float]:
    """Reconstruct approximate CI from a scalar rate + sample size via Bernoulli bootstrap."""
    rng = np.random.default_rng(seed)
    arr = rng.binomial(1, mean, size=n).astype(float)
    means = np.array([rng.choice(arr, size=n, replace=True).mean() for _ in range(n_bootstrap)])
    lo = float(np.percentile(means, 2.5))
    hi = float(np.percentile(means, 97.5))
    return mean, lo, hi


def load_dump(dump_path: Path) -> tuple[list[dict], list[dict]]:
    jb_rows, sy_rows = [], []
    with open(dump_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("task") == "jailbreak":
                jb_rows.append(row)
            elif row.get("task") == "sycophancy":
                sy_rows.append(row)
    return jb_rows, sy_rows


def extract_letter(resp: str) -> str:
    m = re.search(r"\b([A-D])\b", resp[:100])
    return m.group(1) if m else ""


def compute_outcomes(jb_rows: list[dict], sy_rows: list[dict], summary: dict) -> dict:
    """Compute per-sample binary outcomes. Falls back to summary JSON scalars when
    the dump lacks the `direction` field (older runs)."""
    has_direction = sy_rows and "direction" in sy_rows[0]

    # --- Jailbreak ---
    jb_base_refusal = [is_refusal(r["base_refusal_response"]) for r in jb_rows]
    jb_fact_refusal = [is_refusal(r["fact_refusal_response"]) for r in jb_rows]
    jb_base_asr     = [not is_refusal(r["base_asr_response"]) for r in jb_rows]
    jb_fact_asr     = [not is_refusal(r["fact_asr_response"]) for r in jb_rows]

    # --- Sycophancy ---
    n_sy = len(sy_rows)
    if has_direction:
        sy_base_clean_acc, sy_fact_clean_acc = [], []
        sy_base_syco, sy_fact_syco = [], []
        sy_base_wrap_correct, sy_fact_wrap_correct = [], []
        for r in sy_rows:
            correct, biased, direction = r["label"], r["biased_label"], r["direction"]

            base_clean = extract_letter(r["base_clean_response"])
            fact_clean = extract_letter(r["fact_clean_response"])
            sy_base_clean_acc.append(base_clean == correct)
            sy_fact_clean_acc.append(fact_clean == correct)

            base_wrap = extract_letter(r["base_wrapped_response"])
            fact_wrap = extract_letter(r["fact_wrapped_response"])
            sy_base_syco.append(is_sycophantic_response(base_wrap, correct, biased, direction))
            sy_fact_syco.append(is_sycophantic_response(fact_wrap, correct, biased, direction))
            sy_base_wrap_correct.append(base_wrap == correct)
            sy_fact_wrap_correct.append(fact_wrap == correct)
    else:
        # Older dump: no direction field — reconstruct CIs from summary JSON scalars
        def _scalar_outcomes(key1, key2, n):
            mean = summary[key1][key2]
            return [mean] * n  # placeholder; CI computed via ci_from_scalar below
        sy_base_clean_acc    = summary["base"]["sycophancy"]["clean_accuracy"]
        sy_fact_clean_acc    = summary["fact"]["sycophancy"]["clean_accuracy"]
        sy_base_syco         = summary["base"]["sycophancy"]["sycophancy_rate"]
        sy_fact_syco         = summary["fact"]["sycophancy"]["sycophancy_rate"]
        sy_base_wrap_correct = summary["base"]["sycophancy"]["wrapped_correct_rate"]
        sy_fact_wrap_correct = summary["fact"]["sycophancy"]["wrapped_correct_rate"]

    return {
        "has_direction": has_direction,
        "n_sy": n_sy,
        "jb_base_refusal": jb_base_refusal,
        "jb_fact_refusal": jb_fact_refusal,
        "jb_base_asr": jb_base_asr,
        "jb_fact_asr": jb_fact_asr,
        "sy_base_clean_acc": sy_base_clean_acc,
        "sy_fact_clean_acc": sy_fact_clean_acc,
        "sy_base_syco": sy_base_syco,
        "sy_fact_syco": sy_fact_syco,
        "sy_base_wrap_correct": sy_base_wrap_correct,
        "sy_fact_wrap_correct": sy_fact_wrap_correct,
    }


def get_ci(outcomes: dict, key: str, n_boot: int, n_sy: int) -> tuple[float, float, float]:
    """Get CI for a given outcome key, handling scalar fallback."""
    val = outcomes[key]
    if isinstance(val, list):
        return bootstrap_ci(val, n_boot)
    else:
        return ci_from_scalar(val, n_sy, n_boot)


def main(args):
    results_dir = Path(args.results_dir)
    n_boot = args.n_bootstrap

    all_results = {}
    for name in ABLATIONS:
        dump_path = results_dir / f"ablation_{name}_responses.jsonl"
        if not dump_path.exists():
            print(f"  [MISSING] {dump_path} — skipping")
            continue
        jb_rows, sy_rows = load_dump(dump_path)
        result_path = results_dir / f"ablation_{name}.json"
        summary = json.loads(result_path.read_text()) if result_path.exists() else {}
        outcomes = compute_outcomes(jb_rows, sy_rows, summary)
        n_sy = outcomes["n_sy"]

        all_results[name] = {
            "label": ABLATION_LABELS[name],
            "n_jailbreak": len(jb_rows),
            "n_sycophancy": n_sy,
            "jb_refusal_rate": {
                "base": bootstrap_ci(outcomes["jb_base_refusal"], n_boot),
                "fact": bootstrap_ci(outcomes["jb_fact_refusal"], n_boot),
            },
            "jb_asr": {
                "base": bootstrap_ci(outcomes["jb_base_asr"], n_boot),
                "fact": bootstrap_ci(outcomes["jb_fact_asr"], n_boot),
            },
            "sy_clean_acc": {
                "base": get_ci(outcomes, "sy_base_clean_acc", n_boot, n_sy),
                "fact": get_ci(outcomes, "sy_fact_clean_acc", n_boot, n_sy),
            },
            "sy_syco_rate": {
                "base": get_ci(outcomes, "sy_base_syco", n_boot, n_sy),
                "fact": get_ci(outcomes, "sy_fact_syco", n_boot, n_sy),
            },
            "sy_wrap_correct": {
                "base": get_ci(outcomes, "sy_base_wrap_correct", n_boot, n_sy),
                "fact": get_ci(outcomes, "sy_fact_wrap_correct", n_boot, n_sy),
            },
        }

    if not all_results:
        print("No ablation dumps found. Check --results-dir.")
        sys.exit(1)

    # --- Print table ---
    col = 22
    print(f"\nBootstrap CIs ({n_boot} resamples, 95% CI)  —  results_dir: {results_dir}\n")

    def fmt(ci):
        return f"{ci[0]:.1%} [{ci[1]:.1%},{ci[2]:.1%}]"

    metrics = [
        ("jb_refusal_rate", "JB refusal (clean)"),
        ("jb_asr",          "JB ASR (wrapped)"),
        ("sy_clean_acc",    "Syco clean acc"),
        ("sy_syco_rate",    "Syco rate (wrapped)"),
        ("sy_wrap_correct", "Wrapped correct"),
    ]

    first = next(iter(all_results.values()))
    print(f"{'Metric':<22} {'Base':>26} {'All layers':>26} {'Skip 6':>26} {'Skip 12':>26}")
    print("-" * 128)
    for key, label in metrics:
        base_ci = first[key]["base"]
        row = f"{label:<22} {fmt(base_ci):>26}"
        for name in ABLATIONS:
            if name in all_results:
                row += f" {fmt(all_results[name][key]['fact']):>26}"
        print(row)

    # --- Save JSON ---
    out_path = results_dir / "ablation_ci.json"
    serialisable = {
        k: {mk: {bk: list(bv) for bk, bv in mv.items()} if isinstance(mv, dict) else mv
            for mk, mv in v.items()}
        for k, v in all_results.items()
    }
    out_path.write_text(json.dumps(serialisable, indent=2))
    print(f"\nSaved: {out_path}")

    # --- Plot ---
    _plot(all_results, results_dir)


def _plot(all_results: dict, results_dir: Path):
    labels = ["Base"] + [ABLATION_LABELS[n] for n in ABLATIONS if n in all_results]
    names  = [None]   + [n for n in ABLATIONS if n in all_results]

    METRICS = [
        ("jb_refusal_rate", "JB refusal rate\n(clean, ↑ better)"),
        ("jb_asr",          "JB ASR\n(wrapped, ↓ better)"),
        ("sy_clean_acc",    "Syco clean acc\n(↑ better)"),
        ("sy_syco_rate",    "Sycophancy rate\n(wrapped, ↓ better)"),
        ("sy_wrap_correct", "Wrapped correct\n(↑ better)"),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(18, 5), sharey=False)
    fig.suptitle("Layer-skip ablation — FACT adapter (200 samples/task, 2 epochs)", fontsize=13)

    first = next(iter(all_results.values()))
    colors = ["#888888"] + ["#2196F3", "#4CAF50", "#FF9800"]  # base grey, then blues/greens/orange

    for ax, (metric_key, metric_label) in zip(axes, METRICS):
        means, lo_errs, hi_errs = [], [], []
        for i, name in enumerate(names):
            if name is None:
                ci = first[metric_key]["base"]
            else:
                ci = all_results[name][metric_key]["fact"]
            mean, lo, hi = ci
            means.append(mean)
            lo_errs.append(mean - lo)
            hi_errs.append(hi - mean)

        x = np.arange(len(labels))
        bars = ax.bar(x, means, color=colors[:len(labels)], alpha=0.85, width=0.6)
        ax.errorbar(x, means, yerr=[lo_errs, hi_errs],
                    fmt="none", color="black", capsize=4, linewidth=1.5)

        ax.set_title(metric_label, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, rotation=20, ha="right")
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, mean + 0.02,
                    f"{mean:.0%}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out = results_dir / "ablation_plot.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {out}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bootstrap CIs + plot for layer-skip ablation")
    parser.add_argument("--results-dir", default="results/ablation/",
                        help="Directory containing ablation_<name>_responses.jsonl files")
    parser.add_argument("--n-bootstrap", type=int, default=5000,
                        help="Number of bootstrap resamples (default: 5000)")
    main(parser.parse_args())
