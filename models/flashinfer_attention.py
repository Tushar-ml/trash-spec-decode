"""
Llama attention using FlashInfer single-request decode/prefill kernels (CUDA).

Uses contiguous KV tensors compatible with Hugging Face DynamicCache (NHD layout
after permute). Tree verification uses single_prefill_with_kv_cache + custom_mask
(bool True = attend), matching Token Recycling (arXiv:2408.08696).

Requires: pip install flashinfer-python (see requirements-optional.txt). CUDA required.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from transformers.cache_utils import Cache

from .modeling_llama import LlamaSdpaAttention, apply_rotary_pos_emb


def flashinfer_available() -> bool:
    try:
        import flashinfer  # noqa: F401

        return torch.cuda.is_available()
    except ImportError:
        return False


def apply_flash_infer_attention(model: torch.nn.Module) -> None:
    """
    Replace LlamaSdpaAttention layers with LlamaFlashInferAttention (same weights).
    Hugging Face does not register `flash_infer`, so we load as sdpa then swap.
    """
    from .modeling_llama import LlamaForCausalLM

    if not isinstance(model, LlamaForCausalLM):
        raise TypeError("Expected LlamaForCausalLM")
    for layer in model.model.layers:
        old = layer.self_attn
        new = LlamaFlashInferAttention(old.config, layer_idx=old.layer_idx)
        new.load_state_dict(old.state_dict())
        layer.self_attn = new
    model.config._attn_implementation = "flash_infer"


def _resolve_flashinfer_ops():
    """Return (single_decode, single_prefill) callables."""
    import flashinfer as fi

    dec = getattr(fi, "single_decode_with_kv_cache", None)
    pre = getattr(fi, "single_prefill_with_kv_cache", None)
    if dec is None:
        from flashinfer.decode import single_decode_with_kv_cache as dec
    if pre is None:
        from flashinfer.prefill import single_prefill_with_kv_cache as pre
    return dec, pre


def _sdpa_mask_to_bool_2d(
    attn_4d: torch.Tensor, q_len: int, kv_len: int
) -> torch.Tensor:
    """(1,1,q,kv) float additive mask -> (q,kv) bool, True = allow attention."""
    m = attn_4d[0, 0, :q_len, :kv_len]
    return m >= (torch.finfo(m.dtype).min / 2)


class LlamaFlashInferAttention(LlamaSdpaAttention):
    """
    FlashInfer kernels for decode (1 new token) and prefill / tree (custom mask).
    Falls back to SDPA when unsupported (e.g. bsz != 1) or on kernel errors.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        if output_attentions or hidden_states.device.type != "cuda" or bsz != 1:
            return LlamaSdpaAttention.forward(
                self,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        # Rare HF path: full sequence + KV cache without custom mask (recompute)
        if past_key_value is not None and q_len > 1 and attention_mask is None:
            return LlamaSdpaAttention.forward(
                self,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        try:
            single_decode, single_prefill = _resolve_flashinfer_ops()
        except ImportError:
            return LlamaSdpaAttention.forward(
                self,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        kv_len = key_states.shape[-2]
        sm_scale = 1.0 / math.sqrt(self.head_dim)

        k_nhd = key_states[0].permute(1, 0, 2).contiguous()
        v_nhd = value_states[0].permute(1, 0, 2).contiguous()
        q_nhd = query_states[0].permute(1, 0, 2).contiguous()

        if attention_mask is None:
            if q_len == 1 and past_key_value is not None:
                q_dec = q_nhd[0]
                o = single_decode(
                    q_dec,
                    k_nhd,
                    v_nhd,
                    kv_layout="NHD",
                    pos_encoding_mode="NONE",
                    sm_scale=sm_scale,
                )
                attn_output = o.view(1, 1, -1)
            else:
                o = single_prefill(
                    q_nhd,
                    k_nhd,
                    v_nhd,
                    causal=True,
                    kv_layout="NHD",
                    pos_encoding_mode="NONE",
                    sm_scale=sm_scale,
                )
                attn_output = o.view(1, q_len, -1)
        else:
            causal_mask = attention_mask[:, :, :, :kv_len]
            custom_bool = _sdpa_mask_to_bool_2d(causal_mask, q_len, kv_len)
            o = single_prefill(
                q_nhd,
                k_nhd,
                v_nhd,
                causal=False,
                custom_mask=custom_bool,
                kv_layout="NHD",
                pos_encoding_mode="NONE",
                sm_scale=sm_scale,
            )
            attn_output = o.view(1, q_len, -1)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value
