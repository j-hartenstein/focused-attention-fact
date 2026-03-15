"""Shared utilities: token alignment, logging, result serialization."""

from __future__ import annotations

import json
import logging
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

import torch


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )
    return logging.getLogger(name)


def compute_icore(
    clean_ids: torch.Tensor,
    wrapped_ids: torch.Tensor,
) -> tuple[list[int], list[int]]:
    """Find the shared suffix (core query) between clean and wrapped token sequences.

    Returns (clean_indices, wrapped_indices) — parallel index lists into each sequence
    pointing to the shared suffix tokens. The suffix is computed by longest common suffix
    match. Always includes at least the last token (EOS / generation trigger).

    Args:
        clean_ids:   1D tensor of token ids for the clean prompt.
        wrapped_ids: 1D tensor of token ids for the wrapped prompt.

    Returns:
        Tuple of two lists of int indices (into clean_ids and wrapped_ids respectively).
    """
    clean = clean_ids.tolist()
    wrapped = wrapped_ids.tolist()

    # Walk backwards from the end to find the longest common suffix.
    ci, wi = len(clean) - 1, len(wrapped) - 1
    clean_core, wrapped_core = [], []
    while ci >= 0 and wi >= 0 and clean[ci] == wrapped[wi]:
        clean_core.append(ci)
        wrapped_core.append(wi)
        ci -= 1
        wi -= 1

    if not clean_core:
        # No common suffix — fall back to last token only.
        clean_core = [len(clean) - 1]
        wrapped_core = [len(wrapped) - 1]
    else:
        clean_core = clean_core[::-1]
        wrapped_core = wrapped_core[::-1]

    return clean_core, wrapped_core


def _find_subsequence_span(
    sequence: list[int],
    pattern: list[int],
    prefer_last: bool = False,
) -> Optional[tuple[int, int]]:
    """Find [start, end) span of pattern in sequence, or None if not found."""
    if not pattern or len(pattern) > len(sequence):
        return None

    starts = range(len(sequence) - len(pattern), -1, -1) if prefer_last else range(0, len(sequence) - len(pattern) + 1)
    for start in starts:
        if sequence[start:start + len(pattern)] == pattern:
            return start, start + len(pattern)
    return None


def compute_icore_from_core_span(
    clean_ids: torch.Tensor,
    wrapped_ids: torch.Tensor,
    core_ids: torch.Tensor,
) -> Optional[tuple[list[int], list[int]]]:
    """Align I_core using an explicit core token span.

    Returns aligned index lists when the same core token sequence is found in both
    clean and wrapped inputs. Returns None when span alignment fails.
    """
    clean = clean_ids.tolist()
    wrapped = wrapped_ids.tolist()
    core = core_ids.tolist()

    clean_span = _find_subsequence_span(clean, core, prefer_last=False)
    # Wrapped prompts may mention the core text more than once (e.g. instructions + prompt),
    # so prefer the last occurrence, typically nearest the generation point.
    wrapped_span = _find_subsequence_span(wrapped, core, prefer_last=True)

    if clean_span is None or wrapped_span is None:
        return None

    clean_start, clean_end = clean_span
    wrapped_start, wrapped_end = wrapped_span
    clean_core = list(range(clean_start, clean_end))
    wrapped_core = list(range(wrapped_start, wrapped_end))
    if len(clean_core) != len(wrapped_core):
        return None
    return clean_core, wrapped_core


def compute_icore_windowed(
    clean_ids: torch.Tensor,
    wrapped_ids: torch.Tensor,
    min_block_tokens: int = 8,
    min_total_tokens: int = 8,
) -> Optional[tuple[list[int], list[int]]]:
    """Align I_core using exact token-ID matching over multi-token blocks.

    This uses ``SequenceMatcher`` over token IDs and keeps only contiguous matches
    of at least ``min_block_tokens`` tokens. The output is the concatenation of all
    retained aligned blocks in monotonic order.

    Args:
        clean_ids: 1D token IDs for clean prompt.
        wrapped_ids: 1D token IDs for wrapped prompt.
        min_block_tokens: minimum contiguous block length to accept.
        min_total_tokens: minimum total aligned tokens required for success.

    Returns:
        (clean_indices, wrapped_indices), or None if alignment is too small.
    """
    clean = clean_ids.tolist()
    wrapped = wrapped_ids.tolist()

    matcher = SequenceMatcher(None, clean, wrapped, autojunk=False)
    blocks = matcher.get_matching_blocks()

    clean_core: list[int] = []
    wrapped_core: list[int] = []
    for block in blocks:
        if block.size < min_block_tokens:
            continue
        clean_core.extend(range(block.a, block.a + block.size))
        wrapped_core.extend(range(block.b, block.b + block.size))

    if len(clean_core) < min_total_tokens or len(clean_core) != len(wrapped_core):
        return None
    return clean_core, wrapped_core


def save_results(results: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def load_results(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)
