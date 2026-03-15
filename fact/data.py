"""Dataset loading and PromptPairDataset for FACT experiments."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from fact.utils import get_logger

logger = get_logger(__name__)

Task = Literal["jailbreak", "sycophancy", "instruction_following"]


@dataclass
class PromptPair:
    """A single (clean, wrapped) prompt pair with metadata."""
    clean: str       # clean prompt text (xc)
    wrapped: str     # wrapped prompt text (xw)
    task: Task
    label: str = ""         # ground truth (MCQ letter for sycophancy; "refusal" for jailbreak)
    attack: str = ""        # attack strategy (autodan/msj/prefill/human_mt) or bias name
    dataset: str = ""       # originating dataset name
    source: str = ""        # source tag (e.g., raybears_non_cot)
    biased_label: str = ""  # sycophancy: letter referenced by the bias sentence
    direction: str = ""     # sycophancy: "toward" | "against" | "unclear"
    adversarial: bool = True  # sycophancy: False if bias agrees with truth (filter from metrics)
    core_text: str = ""     # jailbreak: query span shared across clean/wrapped variants
    clean_response: str = ""    # model response to clean prompt (for LM loss anchoring)
    wrapped_response: str = ""  # model response to wrapped prompt


@dataclass
class TokenizedPair:
    clean_ids: torch.Tensor          # (T_c,) — prompt + response if available
    wrapped_ids: torch.Tensor        # (T_w,)
    clean_core_indices: list[int]    # Icore positions in clean sequence
    wrapped_core_indices: list[int]  # Icore positions in wrapped sequence
    pair: PromptPair
    response_start_idx: int = 0      # index in clean_ids where response tokens begin
                                     # (0 means no response / entire sequence is prompt)


class PromptPairDataset(Dataset):
    """Dataset of tokenized (clean, wrapped) prompt pairs."""

    def __init__(
        self,
        pairs: list[PromptPair],
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 2000,
        min_alignment_block_tokens: int = 8,
        min_alignment_total_tokens: int = 8,
    ):
        from fact.utils import (
            compute_icore,
            compute_icore_from_core_span,
            compute_icore_windowed,
        )

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.items: list[TokenizedPair] = []
        n_window_aligned = 0
        n_span_aligned = 0
        n_suffix_fallback = 0
        n_window_failed = 0
        n_span_failed = 0

        for pair in pairs:
            # Use same tokenization as forward pass (incl. special tokens e.g. BOS for Llama)
            # so that clean_core_indices / wrapped_core_indices align with model input positions.
            clean_enc = tokenizer(pair.clean, return_tensors="pt")
            wrapped_enc = tokenizer(pair.wrapped, return_tensors="pt")
            clean_ids = clean_enc["input_ids"][0]
            wrapped_ids = wrapped_enc["input_ids"][0]

            # Tokenize clean_response separately; append after I_core alignment.
            resp_ids = None
            if pair.clean_response:
                resp_ids = tokenizer(
                    pair.clean_response,
                    add_special_tokens=False,
                    return_tensors="pt",
                )["input_ids"][0]

            prompt_len = len(clean_ids)
            total_clean_len = prompt_len + (len(resp_ids) if resp_ids is not None else 0)
            if total_clean_len > max_length or len(wrapped_ids) > max_length:
                continue

            # Compute I_core alignment on the prompt-only portion (before response).
            ci, wi = None, None
            aligned = compute_icore_windowed(
                clean_ids,
                wrapped_ids,
                min_block_tokens=min_alignment_block_tokens,
                min_total_tokens=min_alignment_total_tokens,
            )
            if aligned is not None:
                ci, wi = aligned
                n_window_aligned += 1
            else:
                n_window_failed += 1

            if ci is None and wi is None and pair.task == "jailbreak" and pair.core_text.strip():
                core_ids = tokenizer(
                    pair.core_text,
                    add_special_tokens=False,
                    return_tensors="pt",
                )["input_ids"][0]
                if len(core_ids) > 0:
                    span_aligned = compute_icore_from_core_span(clean_ids, wrapped_ids, core_ids)
                    if span_aligned is not None:
                        ci, wi = span_aligned
                        n_span_aligned += 1
                    else:
                        n_span_failed += 1

            if ci is None or wi is None:
                ci, wi = compute_icore(clean_ids, wrapped_ids)
                n_suffix_fallback += 1

            # Append response tokens after alignment (so I_core indices stay valid).
            response_start_idx = 0
            if resp_ids is not None:
                response_start_idx = len(clean_ids)
                clean_ids = torch.cat([clean_ids, resp_ids])

            self.items.append(TokenizedPair(
                clean_ids=clean_ids,
                wrapped_ids=wrapped_ids,
                clean_core_indices=ci,
                wrapped_core_indices=wi,
                pair=pair,
                response_start_idx=response_start_idx,
            ))

        logger.info(
            f"PromptPairDataset: {len(self.items)} pairs "
            f"(dropped {len(pairs) - len(self.items)} overlong)"
        )
        logger.info(
            "I_core alignment: window_aligned=%d, window_failed=%d, "
            "span_aligned=%d, span_failed=%d, suffix_fallback=%d",
            n_window_aligned,
            n_window_failed,
            n_span_aligned,
            n_span_failed,
            n_suffix_fallback,
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> TokenizedPair:
        return self.items[idx]


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def load_jailbreak_pairs(
    data_path: str | Path,
    tokenizer: PreTrainedTokenizerBase,
    n_samples: Optional[int] = None,
    seed: int = 42,
    attacks: Optional[list[str]] = None,
    informative_only: bool = False,
    split: str = "train",
    val_size: int = 200,
) -> list[PromptPair]:
    """Load prepared jailbreak pairs from a JSONL file.

    Format: {"clean": str, "wrapped": str, "attack": str, "core_text": str}
    Optionally includes: {"clean_response": str, "wrapped_response": str,
                          "clean_refused": bool, "wrapped_asr": bool}

    Args:
        attacks: if given, only load pairs with these attack values
                 (e.g. ["autodan", "msj"]).
        informative_only: if True, only load pairs where clean_refused=True
                          and the wrapped prompt was a jailbreak success.
                          Uses ``wrapped_asr_llm`` (LLM judge) if present,
                          falling back to ``wrapped_asr`` (regex heuristic).
        split: "train" or "val". Splits are determined by shuffling all
               eligible pairs with seed 42, reserving the last ``val_size``
               as val. Applied before ``n_samples`` cap.
        val_size: number of pairs to reserve for the val split (default 200).
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Jailbreak data not found at {path}. "
            "Run: python data/scripts/prepare_jailbreaks.py"
        )

    pairs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            attack = obj.get("attack", "")
            if attacks is not None and attack not in attacks:
                continue
            if informative_only:
                if not obj.get("clean_refused", False):
                    continue
                # Prefer LLM-judged ASR if available; fall back to regex heuristic
                wrapped_success = (
                    obj["wrapped_asr_llm"]
                    if "wrapped_asr_llm" in obj
                    else obj.get("wrapped_asr", False)
                )
                if not wrapped_success:
                    continue
            wrapped = obj["wrapped"].lower() if attack == "best_of_n" else obj["wrapped"]
            pairs.append(PromptPair(
                clean=obj["clean"],
                wrapped=wrapped,
                task="jailbreak",
                attack=attack,
                source=obj.get("source", ""),
                core_text=obj.get("core_text", ""),
                clean_response=obj.get("clean_response", ""),
                wrapped_response=obj.get("wrapped_response", ""),
            ))

    # Train/val split: shuffle with fixed seed, reserve last val_size as val.
    split_rng = random.Random(42)
    split_rng.shuffle(pairs)
    if val_size > 0 and len(pairs) > val_size:
        train_pairs, val_pairs = pairs[:-val_size], pairs[-val_size:]
    else:
        train_pairs, val_pairs = pairs, []
    pairs = val_pairs if split == "val" else train_pairs

    if n_samples is not None and split != "val":
        rng = random.Random(seed)
        pairs = rng.sample(pairs, min(n_samples, len(pairs)))

    logger.info(f"Loaded {len(pairs)} jailbreak pairs from {path} (split={split})")
    return pairs


def load_sycophancy_pairs(
    data_path: str | Path,
    tokenizer: PreTrainedTokenizerBase,
    n_samples: Optional[int] = None,
    seed: int = 42,
    adversarial_only: bool = False,
    informative_only: bool = False,
    split: str = "train",
    val_size: int = 200,
) -> list[PromptPair]:
    """Load prepared sycophancy pairs from a JSONL file.

    Format: {"clean": str, "wrapped": str, "label": str, "biased_label": str,
             "direction": str, "adversarial": bool, "source": str}
    Optionally includes: {"clean_response": str, "wrapped_response": str,
                          "clean_correct": bool, "wrapped_sycophantic": bool}

    Args:
        adversarial_only: if True, only load pairs where adversarial=True
                          (bias toward wrong answer, or against correct answer).
                          Use this for training and metric computation.
        informative_only: if True, only load pairs where clean_correct=True
                          and wrapped_sycophantic=True (model was correct on
                          clean but sycophantic on wrapped).
        split: "train" or "val". Splits are determined by shuffling all
               eligible pairs with seed 42, reserving the last ``val_size``
               as val. Applied before ``n_samples`` cap.
        val_size: number of pairs to reserve for the val split (default 200).
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Sycophancy data not found at {path}. "
            "Run: python data/scripts/prepare_sycophancy.py"
        )

    pairs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            adversarial = obj.get("adversarial", True)  # old files default True
            if adversarial_only and not adversarial:
                continue
            if informative_only:
                if not obj.get("clean_correct", False) or not obj.get("wrapped_sycophantic", False):
                    continue
            pairs.append(PromptPair(
                clean=obj["clean"],
                wrapped=obj["wrapped"],
                task="sycophancy",
                label=obj.get("label", ""),
                attack=obj.get("source", ""),   # source plays the role of "attack" tag
                source=obj.get("source", ""),
                biased_label=obj.get("biased_label", ""),
                direction=obj.get("direction", ""),
                adversarial=adversarial,
                clean_response=obj.get("clean_response", ""),
                wrapped_response=obj.get("wrapped_response", ""),
            ))

    # Train/val split: shuffle with fixed seed, reserve last val_size as val.
    split_rng = random.Random(42)
    split_rng.shuffle(pairs)
    if val_size > 0 and len(pairs) > val_size:
        train_pairs, val_pairs = pairs[:-val_size], pairs[-val_size:]
    else:
        train_pairs, val_pairs = pairs, []
    pairs = val_pairs if split == "val" else train_pairs

    if n_samples is not None and split != "val":
        rng = random.Random(seed)
        pairs = rng.sample(pairs, min(n_samples, len(pairs)))

    logger.info(
        f"Loaded {len(pairs)} sycophancy pairs from {path} (split={split})"
        + (" (adversarial_only=True)" if adversarial_only else "")
    )
    return pairs


# ---------------------------------------------------------------------------
# Shared paths and loaders (patching + FACT use the same datasets)
# ---------------------------------------------------------------------------

def load_if_pairs(
    data_path: str | Path,
    n_samples: Optional[int] = None,
    seed: int = 42,
    verified_only: bool = True,
) -> list[PromptPair]:
    """Load instruction-following pairs from the IF dataset.

    These are CE-only samples: clean==wrapped (the wrapped_prompt IS the full
    prompt including the format instruction), so FACT loss is meaningless and
    must be skipped during training.

    Format: {"prompt": str, "wrapped_prompt": str, "response": str,
             "wrapper_id": str, "wrapper_text": str, "verified": bool}
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(
            f"IF dataset not found at {path}. "
            "Run: python data/scripts/build_if_dataset.py"
        )

    pairs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if verified_only and not obj.get("verified", True):
                continue
            wrapped_prompt = obj["wrapped_prompt"]
            response = obj["response"]
            pairs.append(PromptPair(
                clean=wrapped_prompt,
                wrapped=wrapped_prompt,   # same — no adversarial wrapping
                task="instruction_following",
                label=obj.get("wrapper_id", ""),
                attack=obj.get("wrapper_id", ""),
                source="if_dataset",
                clean_response=response,
                wrapped_response=response,
                adversarial=False,        # signals: skip FACT loss
            ))

    if n_samples is not None:
        rng = random.Random(seed)
        pairs = rng.sample(pairs, min(n_samples, len(pairs)))

    logger.info(f"Loaded {len(pairs)} instruction-following pairs from {path}")
    return pairs


# Pre-filtered informative pairs (clean-correct, wrapped-incorrect) for patching and FACT.
# These files lack response fields; use FACT_DATA + informative_only=True for training.
PATCHING_DATA: dict[str, str] = {
    "jailbreak": "data/patching/jailbreaks_informative_160.jsonl",
    "sycophancy": "data/patching/sycophancy_informative_170.jsonl",
}

# All-responses files (include clean_response/wrapped_response for LM loss anchoring).
# Use with informative_only=True in loaders to get the same informative subset.
FACT_DATA: dict[str, str] = {
    "jailbreak": "data/processed/jailbreak_all_responses.jsonl",
    "sycophancy": "data/processed/sycophancy_strong_cues_all_responses.jsonl",
}

IF_DATA = "data/processed/if_dataset.jsonl"

PATCHING_LOADERS = {
    "jailbreak": load_jailbreak_pairs,
    "sycophancy": load_sycophancy_pairs,
}
