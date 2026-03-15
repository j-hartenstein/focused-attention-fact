"""FACT fine-tuning: LoRA setup, L_Inv loss, and training loop."""

from __future__ import annotations

from typing import Optional

import torch
from peft import LoraConfig, get_peft_model, PeftModel
from peft import TaskType
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from fact.data import TokenizedPair
from fact.utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Collate and batching
# ---------------------------------------------------------------------------

def collate_tokenized_pairs(
    items: list[TokenizedPair],
    pad_token_id: int,
    device: Optional[torch.device] = None,
) -> dict[str, torch.Tensor | list[list[int]]]:
    """Collate a list of TokenizedPair into a batch with left-padding.

    Returns dict with:
        clean_ids: (B, max_Tc) long
        wrapped_ids: (B, max_Tw) long
        attention_mask_clean: (B, max_Tc) 1 where valid, 0 where pad
        attention_mask_wrapped: (B, max_Tw)
        clean_core_indices: list of B lists of int
        wrapped_core_indices: list of B lists of int
    """
    max_clean = max(item.clean_ids.shape[0] for item in items)
    max_wrapped = max(item.wrapped_ids.shape[0] for item in items)
    B = len(items)

    clean_ids = torch.full((B, max_clean), pad_token_id, dtype=torch.long)
    wrapped_ids = torch.full((B, max_wrapped), pad_token_id, dtype=torch.long)
    attention_mask_clean = torch.zeros(B, max_clean, dtype=torch.long)
    attention_mask_wrapped = torch.zeros(B, max_wrapped, dtype=torch.long)

    clean_core_indices: list[list[int]] = []
    wrapped_core_indices: list[list[int]] = []
    response_start_indices: list[int] = []

    apply_fact = torch.zeros(B, dtype=torch.bool)

    for i, item in enumerate(items):
        Tc, Tw = item.clean_ids.shape[0], item.wrapped_ids.shape[0]
        pad_offset = max_clean - Tc  # left-padding offset
        clean_ids[i, -Tc:] = item.clean_ids
        wrapped_ids[i, -Tw:] = item.wrapped_ids
        attention_mask_clean[i, -Tc:] = 1
        attention_mask_wrapped[i, -Tw:] = 1
        clean_core_indices.append(item.clean_core_indices)
        wrapped_core_indices.append(item.wrapped_core_indices)
        # Adjust response_start_idx for left-padding
        response_start_indices.append(item.response_start_idx + pad_offset)
        # FACT loss only for adversarial pairs (not CE-only IF samples)
        apply_fact[i] = item.pair.adversarial

    out: dict[str, torch.Tensor | list[list[int]] | list[int]] = {
        "clean_ids": clean_ids,
        "wrapped_ids": wrapped_ids,
        "attention_mask_clean": attention_mask_clean,
        "attention_mask_wrapped": attention_mask_wrapped,
        "clean_core_indices": clean_core_indices,
        "wrapped_core_indices": wrapped_core_indices,
        "response_start_indices": response_start_indices,
        "apply_fact": apply_fact,
    }
    if device is not None:
        out["clean_ids"] = clean_ids.to(device)
        out["wrapped_ids"] = wrapped_ids.to(device)
        out["attention_mask_clean"] = attention_mask_clean.to(device)
        out["attention_mask_wrapped"] = attention_mask_wrapped.to(device)
    return out


# ---------------------------------------------------------------------------
# FACT loss (last row; optional: mask clean non-core, penalize wrapped non-core)
# ---------------------------------------------------------------------------

def compute_fact_loss(
    attentions_clean: tuple[torch.Tensor, ...],
    attentions_wrapped: tuple[torch.Tensor, ...],
    clean_core_indices: list[list[int]],
    wrapped_core_indices: list[list[int]],
    zero_clean_noncore: bool = False,
    renormalize_clean_masked: bool = False,
) -> torch.Tensor:
    """Compute L_Inv over the last attention row (generation trigger).

    Core term: L_core = sum_{l,h} || a^{(c)}_{l,h}[-1, I_core] - a^{(w)}_{l,h}[-1, I_core] ||_2^2
    (optionally with clean row zeroed outside I_core and renormalized).
    """
    n_layers = len(attentions_clean)
    if n_layers == 0 or len(attentions_wrapped) != n_layers:
        return torch.tensor(0.0, device=attentions_clean[0].device, dtype=attentions_clean[0].dtype)

    B = attentions_clean[0].shape[0]
    total_loss = torch.tensor(0.0, device=attentions_clean[0].device, dtype=attentions_clean[0].dtype)
    n_terms = 0

    for layer_idx in range(n_layers):
        a_c = attentions_clean[layer_idx]   # (B, H, T_c, T_c)
        a_w = attentions_wrapped[layer_idx]  # (B, H, T_w, T_w)

        for b in range(B):
            ci = clean_core_indices[b]
            wi = wrapped_core_indices[b]
            if len(ci) == 0 or len(wi) != len(ci):
                continue
            # Last row (query position = last token): (H, T)
            row_c = a_c[b, :, -1, :]  # (H, T_c)
            row_w = a_w[b, :, -1, :]  # (H, T_w)

            if zero_clean_noncore:
                # Zero out clean attention at non-I_core key positions
                row_c_masked = row_c.clone()
                mask = torch.ones(row_c.shape[1], dtype=torch.bool, device=row_c.device)
                mask[ci] = False
                row_c_masked[:, mask] = 0.0
                if renormalize_clean_masked:
                    row_sum = row_c_masked.sum(dim=-1, keepdim=True).clamp(min=1e-9)
                    row_c_masked = row_c_masked / row_sum
                row_c = row_c_masked

            # Core term: distance on I_core
            core_c = row_c[:, ci]  # (H, n_core)
            core_w = row_w[:, wi]   # (H, n_core)
            diff_sq = (core_c - core_w).pow(2).sum()
            total_loss = total_loss + diff_sq
            n_terms += 1

    if n_terms == 0:
        return torch.tensor(0.0, device=attentions_clean[0].device, dtype=attentions_clean[0].dtype)
    return total_loss / n_terms


def compute_fact_loss_breakdown(
    attentions_clean: tuple[torch.Tensor, ...],
    attentions_wrapped: tuple[torch.Tensor, ...],
    clean_core_indices: list[list[int]],
    wrapped_core_indices: list[list[int]],
    zero_clean_noncore: bool = False,
    renormalize_clean_masked: bool = False,
) -> tuple[float, list[float], list[float]]:
    """Compute L_Inv with per-layer and per-pair breakdown for evaluation plots.

    Returns:
        mean_loss: scalar mean over all (layer, pair) terms.
        per_layer_means: list of length n_layers; mean L_Inv over pairs for that layer.
        per_pair_means: list of length B; mean L_Inv over layers for that pair.
    """
    n_layers = len(attentions_clean)
    if n_layers == 0 or len(attentions_wrapped) != n_layers:
        return 0.0, [], []

    B = attentions_clean[0].shape[0]
    per_layer_sum: list[float] = [0.0] * n_layers
    per_layer_count: list[int] = [0] * n_layers
    per_pair_sum: list[float] = [0.0] * B
    per_pair_count: list[int] = [0] * B
    total_sum = 0.0
    total_count = 0

    for layer_idx in range(n_layers):
        a_c = attentions_clean[layer_idx]
        a_w = attentions_wrapped[layer_idx]

        for b in range(B):
            ci = clean_core_indices[b]
            wi = wrapped_core_indices[b]
            if len(ci) == 0 or len(wi) != len(ci):
                continue

            row_c = a_c[b, :, -1, :].detach().float()
            row_w = a_w[b, :, -1, :].detach().float()

            if zero_clean_noncore:
                row_c_masked = row_c.clone()
                mask = torch.ones(row_c.shape[1], dtype=torch.bool, device=row_c.device)
                mask[ci] = False
                row_c_masked[:, mask] = 0.0
                if renormalize_clean_masked:
                    row_sum = row_c_masked.sum(dim=-1, keepdim=True).clamp(min=1e-9)
                    row_c_masked = row_c_masked / row_sum
                row_c = row_c_masked

            core_c = row_c[:, ci]
            core_w = row_w[:, wi]
            diff_sq = (core_c - core_w).pow(2).sum().item()

            per_layer_sum[layer_idx] += diff_sq
            per_layer_count[layer_idx] += 1
            per_pair_sum[b] += diff_sq
            per_pair_count[b] += 1
            total_sum += diff_sq
            total_count += 1

    mean_loss = total_sum / total_count if total_count else 0.0
    per_layer_means = [
        per_layer_sum[l] / per_layer_count[l] if per_layer_count[l] else 0.0
        for l in range(n_layers)
    ]
    per_pair_means = [
        per_pair_sum[b] / per_pair_count[b] if per_pair_count[b] else 0.0
        for b in range(B)
    ]
    return mean_loss, per_layer_means, per_pair_means


# ---------------------------------------------------------------------------
# LoRA setup
# ---------------------------------------------------------------------------

def get_lora_model(
    model: PreTrainedModel,
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: Optional[list[str]] = None,
) -> PeftModel:
    """Wrap the model with LoRA adapters using PEFT.

    Default target_modules for Llama: q_proj, v_proj.
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]

    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    return peft_model


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_fact(
    model: PreTrainedModel | PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    train_dataloader: DataLoader,
    *,
    epochs: int = 3,
    lr: float = 2e-5,
    adapter_output_path: str,
    device: torch.device,
    log_interval: int = 10,
    zero_clean_noncore: bool = False,
    renormalize_clean_masked: bool = False,
    fact_loss_weight: float = 1.0,
    max_grad_norm: float = 1.0,
    fact_layers: Optional[list[int]] = None,
    layer_eval_batches: int = 0,
) -> None:
    """Run FACT fine-tuning: total_loss = lm_loss + fact_loss_weight * L_Inv.

    LM loss is next-token prediction on the **clean** sequence (capability
    anchor). L_Inv is the last-row attention invariance over I_core between
    clean and wrapped attention patterns.

    Args:
        fact_layers: If provided, only include these layer indices in the
            FACT loss (e.g. [8, 9, ..., 31] to skip early layers). If None,
            all layers are used.
        max_grad_norm: Maximum gradient norm for clipping (default 1.0).
        layer_eval_batches: If > 0, run a no-grad eval pass over this many
            batches at the end of each epoch using compute_fact_loss_breakdown
            across ALL layers (regardless of fact_layers). Results saved to
            layer_history.json in adapter_output_path.
    """
    model.train()
    model.gradient_checkpointing_enable()
    logger.info("Gradient checkpointing enabled")

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=lr,
    )

    history: list[dict[str, float]] = []
    layer_history: list[dict] = []

    step = 0
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for batch in train_dataloader:
            clean_ids = batch["clean_ids"].to(device)
            wrapped_ids = batch["wrapped_ids"].to(device)
            attention_mask_clean = batch["attention_mask_clean"].to(device)
            attention_mask_wrapped = batch["attention_mask_wrapped"].to(device)
            clean_core_indices = batch["clean_core_indices"]
            wrapped_core_indices = batch["wrapped_core_indices"]
            response_start_indices = batch["response_start_indices"]
            apply_fact_mask = batch["apply_fact"].to(device)  # (B,) bool

            # Labels for LM loss on *clean* sequence: anchor model capabilities
            # by training next-token prediction on the response portion only.
            # Mask padding AND prompt tokens (everything before response_start_idx).
            clean_labels = clean_ids.clone()
            clean_labels[attention_mask_clean == 0] = -100
            for b_idx in range(clean_ids.shape[0]):
                rs = response_start_indices[b_idx]
                if rs > 0:
                    clean_labels[b_idx, :rs] = -100

            optimizer.zero_grad()

            # --- Pass 1: LM loss on clean (no attention needed) ---
            out_clean_lm = model(
                input_ids=clean_ids,
                attention_mask=attention_mask_clean,
                labels=clean_labels,
            )
            lm_loss = out_clean_lm.loss

            # --- Passes 2 & 3: FACT loss (only for adversarial samples) ---
            fact_indices = apply_fact_mask.nonzero(as_tuple=True)[0]
            fact_loss = torch.tensor(0.0, device=device)

            if len(fact_indices) > 0:
                fact_clean_ids = clean_ids[fact_indices]
                fact_wrapped_ids = wrapped_ids[fact_indices]
                fact_mask_clean = attention_mask_clean[fact_indices]
                fact_mask_wrapped = attention_mask_wrapped[fact_indices]
                fact_ci = [clean_core_indices[i] for i in fact_indices.tolist()]
                fact_wi = [wrapped_core_indices[i] for i in fact_indices.tolist()]

                # --- Pass 2: Clean attention targets (no grad, just reference) ---
                with torch.no_grad():
                    out_clean_attn = model(
                        input_ids=fact_clean_ids,
                        attention_mask=fact_mask_clean,
                        output_attentions=True,
                    )
                clean_attentions = tuple(a.detach() for a in out_clean_attn.attentions)

                # --- Pass 3: Wrapped attention (grad needed for FACT loss) ---
                out_wrapped = model(
                    input_ids=fact_wrapped_ids,
                    attention_mask=fact_mask_wrapped,
                    output_attentions=True,
                )

                if out_wrapped.attentions is None:
                    raise RuntimeError(
                        "Model did not return attentions. Use attn_implementation='eager' when loading."
                    )

                # Filter to selected layers for FACT loss
                clean_attn_for_fact = clean_attentions
                wrapped_attn_for_fact = out_wrapped.attentions
                if fact_layers is not None:
                    clean_attn_for_fact = tuple(clean_attentions[l] for l in fact_layers)
                    wrapped_attn_for_fact = tuple(out_wrapped.attentions[l] for l in fact_layers)

                fact_loss = compute_fact_loss(
                    clean_attn_for_fact,
                    wrapped_attn_for_fact,
                    fact_ci,
                    fact_wi,
                    zero_clean_noncore=zero_clean_noncore,
                    renormalize_clean_masked=renormalize_clean_masked,
                )

            loss = lm_loss + fact_loss_weight * fact_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad),
                max_grad_norm,
            )
            optimizer.step()

            _lm = lm_loss.item()
            _fact = fact_loss.item()
            _total = loss.item()
            history.append({"step": step, "epoch": epoch, "loss": _total, "lm_loss": _lm, "fact_loss": _fact})

            epoch_loss += _total
            n_batches += 1
            step += 1
            if log_interval and step % log_interval == 0:
                logger.info(
                    f"  step {step}  loss={_total:.4f}  lm={_lm:.4f}  fact={_fact:.4f}"
                )

        avg_loss = epoch_loss / max(n_batches, 1)
        logger.info(f"Epoch {epoch + 1}/{epochs}  avg_loss={avg_loss:.6f}")

        # --- Per-layer eval pass (optional) ---
        if layer_eval_batches > 0:
            model.eval()
            n_layers = model.config.num_hidden_layers
            layer_sums = [0.0] * n_layers
            layer_counts = [0] * n_layers
            n_eval = 0
            for batch in train_dataloader:
                if n_eval >= layer_eval_batches:
                    break
                fact_indices = batch["apply_fact"].nonzero(as_tuple=True)[0]
                if len(fact_indices) == 0:
                    continue
                clean_ids = batch["clean_ids"][fact_indices].to(device)
                wrapped_ids = batch["wrapped_ids"][fact_indices].to(device)
                mask_clean = batch["attention_mask_clean"][fact_indices].to(device)
                mask_wrapped = batch["attention_mask_wrapped"][fact_indices].to(device)
                ci = [batch["clean_core_indices"][i] for i in fact_indices.tolist()]
                wi = [batch["wrapped_core_indices"][i] for i in fact_indices.tolist()]
                with torch.no_grad():
                    attn_clean = model(input_ids=clean_ids, attention_mask=mask_clean,
                                       output_attentions=True).attentions
                    attn_wrapped = model(input_ids=wrapped_ids, attention_mask=mask_wrapped,
                                         output_attentions=True).attentions
                _, per_layer, _ = compute_fact_loss_breakdown(
                    attn_clean, attn_wrapped, ci, wi,
                    zero_clean_noncore=zero_clean_noncore,
                    renormalize_clean_masked=renormalize_clean_masked,
                )
                for l, v in enumerate(per_layer):
                    if v > 0:
                        layer_sums[l] += v
                        layer_counts[l] += 1
                n_eval += 1
            layer_means = [
                layer_sums[l] / layer_counts[l] if layer_counts[l] > 0 else 0.0
                for l in range(n_layers)
            ]
            layer_history.append({"epoch": epoch, "per_layer_means": layer_means})
            logger.info(f"  Layer eval ({n_eval} batches): "
                        f"min={min(layer_means):.4f} max={max(layer_means):.4f} "
                        f"mean={sum(layer_means)/len(layer_means):.4f}")
            model.train()

    # Save adapter (and tokenizer for easy loading later)
    model.save_pretrained(adapter_output_path)
    tokenizer.save_pretrained(adapter_output_path)
    logger.info(f"Adapter and tokenizer saved to {adapter_output_path}")

    # Save training history for plotting
    import json, os
    history_path = os.path.join(adapter_output_path, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f)
    logger.info(f"Training history saved to {history_path} ({len(history)} steps)")

    if layer_history:
        layer_history_path = os.path.join(adapter_output_path, "layer_history.json")
        with open(layer_history_path, "w") as f:
            json.dump(layer_history, f)
        logger.info(f"Layer history saved to {layer_history_path}")
