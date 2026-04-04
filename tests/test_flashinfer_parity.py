"""
FlashInfer vs SDPA parity on a tiny Llama (CUDA + flashinfer-python only).
"""

import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

cuda_available = torch.cuda.is_available()
try:
    import flashinfer  # noqa: F401

    flashinfer_installed = True
except ImportError:
    flashinfer_installed = False

run_flashinfer = cuda_available and flashinfer_installed

pytestmark = pytest.mark.skipif(
    not run_flashinfer,
    reason="requires CUDA and flashinfer-python (pip install -r requirements-optional.txt)",
)


@pytest.fixture
def tiny_llama():
    from transformers.models.llama.configuration_llama import LlamaConfig

    from models.flashinfer_attention import apply_flash_infer_attention
    from models.modeling_llama import LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
    )
    m_sdpa = LlamaForCausalLM(config).half().cuda()
    m_fi = LlamaForCausalLM(config).half().cuda()
    m_fi.load_state_dict(m_sdpa.state_dict())
    apply_flash_infer_attention(m_fi)
    m_sdpa.eval()
    m_fi.eval()
    return m_sdpa, m_fi


@torch.inference_mode()
def test_decode_parity_one_token(tiny_llama):
    m_sdpa, m_fi = tiny_llama
    input_ids = torch.randint(0, 256, (1, 1), device="cuda", dtype=torch.long)
    o1 = m_sdpa(input_ids, use_cache=True)
    o2 = m_fi(input_ids, use_cache=True)
    assert torch.allclose(o1.logits, o2.logits, rtol=2e-2, atol=2e-2)


@torch.inference_mode()
def test_prefill_parity(tiny_llama):
    m_sdpa, m_fi = tiny_llama
    input_ids = torch.randint(0, 256, (1, 12), device="cuda", dtype=torch.long)
    o1 = m_sdpa(input_ids, use_cache=True)
    o2 = m_fi(input_ids, use_cache=True)
    assert torch.allclose(o1.logits, o2.logits, rtol=2e-2, atol=2e-2)


@torch.inference_mode()
def test_tree_mask_parity(tiny_llama):
    """Draft-style forward with 4D additive mask (two draft queries)."""
    m_sdpa, m_fi = tiny_llama
    past_len = 4
    draft_len = 2
    input_ids = torch.randint(0, 256, (1, past_len + draft_len), device="cuda", dtype=torch.long)
    prefix = input_ids[:, :past_len]
    draft = input_ids[:, past_len:]

    o_ctx = m_sdpa(prefix, use_cache=True)
    past = o_ctx.past_key_values

    min_v = torch.finfo(torch.float16).min
    # Allow full prefix + causal within draft: simplified mask for test
    attn = torch.full((1, 1, draft_len, past_len + draft_len), min_v, device="cuda", dtype=torch.float16)
    for qi in range(draft_len):
        attn[0, 0, qi, : past_len + qi + 1] = 0.0

    d1 = m_sdpa(
        draft,
        past_key_values=past,
        attention_mask=attn,
        use_cache=False,
    )
    o_ctx2 = m_fi(prefix, use_cache=True)
    past2 = o_ctx2.past_key_values
    d2 = m_fi(
        draft,
        past_key_values=past2,
        attention_mask=attn,
        use_cache=False,
    )
    assert torch.allclose(d1.logits, d2.logits, rtol=3e-2, atol=3e-2)
