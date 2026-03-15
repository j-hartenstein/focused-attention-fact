"""Model loading and attention hook utilities for FACT experiments."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from fact.utils import get_logger

logger = get_logger(__name__)

DEFAULT_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
FALLBACK_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def load_model_and_tokenizer(
    model_name: str = DEFAULT_MODEL,
    device: str = "auto",
    torch_dtype: torch.dtype = torch.bfloat16,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Decoder-only generation with batched padding should use left padding.
    tokenizer.padding_side = "left"

    model_kwargs = {
        "torch_dtype": torch_dtype,
        "attn_implementation": "eager",  # needed for attention weight access
    }
    if device == "auto":
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if device != "auto":
        model = model.to(device)
    model.eval()
    logger.info(f"Loaded {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params")
    return model, tokenizer


def load_model_with_adapter(
    base_model_name: str,
    adapter_path: str,
    device: str = "auto",
    torch_dtype: torch.dtype = torch.bfloat16,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load base model and apply a saved PEFT adapter. Tokenizer from adapter dir if present."""
    from peft import PeftModel

    model, tokenizer = load_model_and_tokenizer(
        model_name=base_model_name,
        device=device,
        torch_dtype=torch_dtype,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    # Prefer tokenizer from adapter dir (saved at train time)
    import os
    if os.path.isfile(os.path.join(adapter_path, "tokenizer_config.json")):
        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    logger.info(f"Loaded adapter from {adapter_path}")
    return model, tokenizer


def get_attention_modules(model: PreTrainedModel) -> list[nn.Module]:
    """Return a list of attention sub-modules in layer order.

    Works for Qwen2 and Llama-3 architectures, both of which expose
    model.model.layers[i].self_attn.
    """
    layers = model.model.layers
    return [layer.self_attn for layer in layers]


def num_layers(model: PreTrainedModel) -> int:
    return len(model.model.layers)


# ---------------------------------------------------------------------------
# Hook utilities
# ---------------------------------------------------------------------------

AttentionWeights = dict[int, torch.Tensor]  # layer_idx -> (batch, heads, tgt, src)


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads to match number of attention heads (for GQA). Shape: (B, num_kv_heads, T, head_dim) -> (B, num_heads, T, head_dim)."""
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


@contextmanager
def capture_attention_weights(model: PreTrainedModel):
    """Context manager that captures attention weights from a forward pass.

    Yields a dict that will be populated with attention tensors keyed by layer index
    after the forward pass completes.

    Usage:
        with capture_attention_weights(model) as attn:
            outputs = model(input_ids=..., output_attentions=True)
        # attn[layer_idx] is now populated
    """
    captured: AttentionWeights = {}
    hooks = []

    attn_modules = get_attention_modules(model)
    for layer_idx, attn_module in enumerate(attn_modules):
        def make_hook(idx):
            def hook(module, input, output):
                # HuggingFace attention modules return (attn_output, attn_weights, ...)
                # when output_attentions=True; attn_weights shape: (B, H, T, T)
                if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                    captured[idx] = output[1].detach().cpu()
            return hook
        hooks.append(attn_module.register_forward_hook(make_hook(layer_idx)))

    try:
        yield captured
    finally:
        for h in hooks:
            h.remove()


@contextmanager
def patch_attention_weights(
    model: PreTrainedModel,
    clean_attn: AttentionWeights,
    patch_layers: list[int],
    clean_core_indices: list[int],
    wrapped_core_indices: list[int],
    last_position_only: bool = False,
    last_position_noncore_mode: str = "zero",
    last_position_noncore_epsilon: float = 1e-5,
    last_position_noncore_beta: float = 0.01,
    full_matrix_noncore_mode: str = "zero",
    full_matrix_noncore_beta: float = 0.01,
):
    """Context manager that patches attention weights during a wrapped forward pass.

    For each layer in `patch_layers`, zeroes attention for wrapped tokens outside
    Icore and replaces core-to-core attention weights with those from the clean pass.

    The patch is applied to the *attention output* after softmax — we intercept the
    output of the attention module's softmax via a forward hook and override the
    attention weights slice corresponding to Icore before they propagate.

    Args:
        model:                 The model being run.
        clean_attn:            Captured attention from the clean forward pass.
        patch_layers:          List of layer indices to patch.
        clean_core_indices:    Token positions of Icore in the clean sequence.
        wrapped_core_indices:  Token positions of Icore in the wrapped sequence.
        last_position_only:    If True, only patch the attention row of the last
                               prefill token (the <assistant> / generation trigger).
                               This is a more targeted intervention: instead of
                               forcing attention invariance across all positions,
                               we only fix what the generation trigger attends to.
                               The patched row is renormalized so weights sum to 1.
        last_position_noncore_mode:
                               Behavior for non-core key positions in the patched row
                               when `last_position_only=True`:
                               - "zero": set non-core keys to 0.
                               - "keep": keep wrapped row values for non-core keys.
                               - "epsilon": set non-core keys to `last_position_noncore_epsilon`.
                               - "beta": set non-core keys adaptively so total non-core
                                 mass targets `last_position_noncore_beta` before
                                 final renormalization (uniform across non-core keys).
        last_position_noncore_epsilon:
                               Fill value used when mode is "epsilon".
        last_position_noncore_beta:
                               Target fraction of mass assigned to non-core keys in
                               "beta" mode (must satisfy 0 <= beta < 1).
        full_matrix_noncore_mode:
                               Behavior for full-matrix patching (when
                               `last_position_only=False`) on entries that were
                               previously zeroed:
                               - "zero": keep them at 0 (current behavior).
                               - "beta": fill them with an adaptive epsilon per row.
        full_matrix_noncore_beta:
                               Target non-core mass used in full-matrix "beta" mode
                               (must satisfy 0 <= beta < 1).
    """
    hooks = []
    attn_modules = get_attention_modules(model)

    for layer_idx in patch_layers:
        if layer_idx not in clean_attn:
            continue
        attn_module = attn_modules[layer_idx]
        src_weights = clean_attn[layer_idx]  # (B, H, T_clean, T_clean)

        def make_patch_hook(idx, src):
            def hook(module, args, kwargs, output):
                if not (isinstance(output, tuple) and len(output) >= 2 and output[1] is not None):
                    return output
                attn_out, attn_w = output[0], output[1]
                # attn_w: (B, H, T_wrap, T_wrap) on prefill, (B, H, 1, T_wrap) on decode.
                # Skip decode steps — we only patch the prefill, and last_position_only
                # mode has no meaningful action on single-token decode steps.
                hidden_states = kwargs.get("hidden_states") if kwargs else None
                is_prefill = hidden_states is not None and hidden_states.shape[1] == attn_w.shape[3]
                if not is_prefill:
                    return output

                patched_w = attn_w.clone()
                src_dev = src.to(attn_w.device)
                wrapped_q_len = patched_w.shape[2]
                wrapped_k_len = patched_w.shape[3]

                if last_position_only:
                    # Only patch the last query position (generation trigger token).
                    # Replace its attention row with the clean last-position row,
                    # remapped from clean key positions to wrapped key positions.
                    # Non-core key behavior depends on `last_position_noncore_mode`,
                    # then the row is renormalized so attention weights sum to 1.
                    clean_last_q = src_dev.shape[2] - 1   # last clean query position
                    wrapped_last_q = wrapped_q_len - 1    # last wrapped query position

                    if last_position_noncore_mode == "zero":
                        new_row = torch.zeros(
                            patched_w.shape[0], patched_w.shape[1], 1, wrapped_k_len,
                            dtype=patched_w.dtype, device=patched_w.device,
                        )
                    elif last_position_noncore_mode == "keep":
                        new_row = patched_w[:, :, wrapped_last_q : wrapped_last_q + 1, :].clone()
                    elif last_position_noncore_mode == "epsilon":
                        eps = float(last_position_noncore_epsilon)
                        if eps < 0:
                            raise ValueError(
                                "last_position_noncore_epsilon must be non-negative, "
                                f"got {last_position_noncore_epsilon}"
                            )
                        new_row = torch.full(
                            (patched_w.shape[0], patched_w.shape[1], 1, wrapped_k_len),
                            fill_value=eps,
                            dtype=patched_w.dtype,
                            device=patched_w.device,
                        )
                    elif last_position_noncore_mode == "beta":
                        beta = float(last_position_noncore_beta)
                        if beta < 0.0 or beta >= 1.0:
                            raise ValueError(
                                "last_position_noncore_beta must satisfy 0 <= beta < 1, "
                                f"got {last_position_noncore_beta}"
                            )
                        new_row = torch.zeros(
                            patched_w.shape[0], patched_w.shape[1], 1, wrapped_k_len,
                            dtype=patched_w.dtype, device=patched_w.device,
                        )
                    else:
                        raise ValueError(
                            "last_position_noncore_mode must be one of "
                            f"['zero', 'keep', 'epsilon', 'beta'], got {last_position_noncore_mode}"
                        )

                    # Fill core key positions from clean.
                    wrapped_core_set = set()
                    for src_c, wrap_c in zip(clean_core_indices, wrapped_core_indices):
                        if src_c < src_dev.shape[3] and wrap_c < wrapped_k_len:
                            new_row[:, :, 0, wrap_c] = src_dev[:, :, clean_last_q, src_c]
                            wrapped_core_set.add(wrap_c)

                    if last_position_noncore_mode == "beta":
                        noncore_idx = [k for k in range(wrapped_k_len) if k not in wrapped_core_set]
                        n_noncore = len(noncore_idx)
                        if n_noncore > 0 and beta > 0.0:
                            # Let C be the per-row core mass before normalization.
                            # Set each non-core key to eps = beta*C / ((1-beta)*N_noncore),
                            # so non-core mass targets beta after renormalization.
                            core_mass = new_row.sum(dim=-1, keepdim=True)  # (B,H,1,1)
                            eps = (beta / (1.0 - beta)) * core_mass / float(n_noncore)
                            idx = torch.tensor(noncore_idx, dtype=torch.long, device=patched_w.device)
                            new_row[:, :, 0, idx] = eps.squeeze(-1).squeeze(-1).unsqueeze(-1)

                    # Renormalize so the row sums to 1 (preserves attention output scale).
                    row_sum = new_row.sum(dim=-1, keepdim=True).clamp(min=1e-6)
                    new_row = new_row / row_sum
                    patched_w[:, :, wrapped_last_q : wrapped_last_q + 1, :] = new_row
                else:
                    # Full-matrix patch: zero/fill non-core KEY columns only, and
                    # replace core-to-core weights with clean values across all query
                    # positions. Query rows are kept active.
                    wrapped_k_core = [w for w in wrapped_core_indices if w < wrapped_k_len]

                    if wrapped_k_core:
                        k_mask = torch.zeros(wrapped_k_len, dtype=torch.bool, device=patched_w.device)
                        k_mask[wrapped_k_core] = True
                        patched_w[:, :, :, ~k_mask] = 0.0
                    else:
                        patched_w[:, :, :, :] = 0.0

                    for ci, wi in zip(clean_core_indices, wrapped_core_indices):
                        if ci < src_dev.shape[2] and wi < patched_w.shape[2]:
                            for src_c, wrap_c in zip(clean_core_indices, wrapped_core_indices):
                                if src_c < src_dev.shape[3] and wrap_c < patched_w.shape[3]:
                                    patched_w[:, :, wi, wrap_c] = src_dev[:, :, ci, src_c]

                    if full_matrix_noncore_mode == "beta":
                        beta = float(full_matrix_noncore_beta)
                        if beta < 0.0 or beta >= 1.0:
                            raise ValueError(
                                "full_matrix_noncore_beta must satisfy 0 <= beta < 1, "
                                f"got {full_matrix_noncore_beta}"
                            )
                        if wrapped_k_core:
                            noncore_mask = ~k_mask
                        else:
                            noncore_mask = torch.ones(
                                wrapped_k_len, dtype=torch.bool, device=patched_w.device
                            )
                        n_noncore = int(noncore_mask.sum().item())
                        if n_noncore > 0 and beta > 0.0:
                            # Per-row adaptive epsilon for non-core columns:
                            # eps = beta * C / ((1-beta) * N_noncore), where C is
                            # current row core mass (before final renorm).
                            core_mass = patched_w.sum(dim=-1, keepdim=True)  # (B,H,T,1)
                            core_mass_ref = torch.where(
                                core_mass > 0,
                                core_mass,
                                torch.ones_like(core_mass),
                            )
                            eps = (beta / (1.0 - beta)) * core_mass_ref / float(n_noncore)
                            idx = torch.nonzero(noncore_mask, as_tuple=False).squeeze(-1)
                            patched_w[:, :, :, idx] = eps.expand(
                                patched_w.shape[0], patched_w.shape[1], patched_w.shape[2], n_noncore
                            )
                    elif full_matrix_noncore_mode != "zero":
                        raise ValueError(
                            "full_matrix_noncore_mode must be one of "
                            f"['zero', 'beta'], got {full_matrix_noncore_mode}"
                        )

                    # Keep rows normalized in full-matrix modes.
                    row_sum = patched_w.sum(dim=-1, keepdim=True).clamp(min=1e-6)
                    patched_w = patched_w / row_sum

                # Recompute attention output from patched weights so the residual stream
                # is affected. We are on prefill (checked above), so hidden_states covers
                # all key positions and we can build full value_states.
                input_shape = hidden_states.shape[:-1]
                batch, seq_len = hidden_states.shape[0], hidden_states.shape[1]
                head_dim = getattr(module, "head_dim", None)
                if head_dim is None and hasattr(module, "config"):
                    head_dim = getattr(module.config, "head_dim", None) or (
                        module.config.hidden_size // module.config.num_attention_heads
                    )
                hidden_shape = (batch, seq_len, -1, head_dim)
                value_states = (
                    module.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                )
                n_rep = getattr(module, "num_key_value_groups", 1)
                value_states = _repeat_kv(value_states, n_rep)
                attn_output_pre_o = torch.matmul(patched_w, value_states)
                attn_output_pre_o = attn_output_pre_o.transpose(1, 2).contiguous()
                attn_output_pre_o = attn_output_pre_o.reshape(*input_shape, -1).contiguous()
                attn_out = module.o_proj(attn_output_pre_o)
                return (attn_out, patched_w) + output[2:]
            return hook

        hooks.append(
            attn_module.register_forward_hook(
                make_patch_hook(layer_idx, src_weights), with_kwargs=True
            )
        )

    try:
        yield
    finally:
        for h in hooks:
            h.remove()


def layer_subsets(n_layers: int) -> dict[str, list[int]]:
    """Return the 4 coarse patching conditions."""
    third = n_layers // 3
    return {
        "all": list(range(n_layers)),
        "early": list(range(0, third)),
        "mid": list(range(third, 2 * third)),
        "late": list(range(2 * third, n_layers)),
    }
