# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""A tiny randomly-initialised DeepSeek-V4 for the multi-stream lens tests.

Runs the *real* ``transformers.models.deepseek_v4`` modeling code — hyper-
connections, the ``hc_head`` collapse, sparse MoE and the compressed-attention
branch are all the production implementations, only narrow. That is the point:
the rectangular-Jacobian path has to be validated against the architecture it
will actually be fitted on, not a stand-in.

Everything is CPU / fp32 and small enough for the suite to stay fast; the
config is the one from the research plan's Phase 0 gate, plus the head/expert
widths scaled down to match.
"""

from __future__ import annotations

import torch
from transformers.models.deepseek_v4 import DeepseekV4Config, DeepseekV4ForCausalLM

from jlens.hf import DeepseekV4LensModel, from_hf

from .tiny import _ByteTokenizer

#: Streams, width and depth the tests assert against.
HC_MULT = 4
HIDDEN_SIZE = 64
N_LAYERS = 3
VOCAB_SIZE = 256


def tiny_v4_config(**overrides) -> DeepseekV4Config:
    """The Phase 0 config: 3 layers x 64 hidden x 4 hyper-connection streams.

    The attention/expert sub-widths (``q_lora_rank``, ``o_lora_rank``, the
    indexer dims, ``moe_intermediate_size``) are shrunk in proportion; the
    defaults are sized for the 4096-wide real model and would dominate runtime
    here. Everything else — the mHC mapping, the Sinkhorn projection, the
    per-layer attention and MoE schedules — is left at its default so the
    layer types match the real checkpoint's first blocks.
    """
    kwargs = dict(
        num_hidden_layers=N_LAYERS,
        hidden_size=HIDDEN_SIZE,
        hc_mult=HC_MULT,
        n_routed_experts=8,
        num_experts_per_tok=2,
        num_attention_heads=4,
        head_dim=32,
        vocab_size=VOCAB_SIZE,
        num_key_value_heads=1,
        q_lora_rank=32,
        o_groups=2,
        o_lora_rank=16,
        moe_intermediate_size=32,
        n_shared_experts=1,
        index_n_heads=2,
        index_head_dim=16,
        index_topk=8,
        max_position_embeddings=256,
    )
    kwargs.update(overrides)
    return DeepseekV4Config(**kwargs)


def tiny_v4_lens_model(seed: int = 0, **config_overrides) -> DeepseekV4LensModel:
    """A randomly-initialised tiny V4 wrapped by :func:`jlens.hf.from_hf`.

    fp32 throughout: the finite-difference gate differences the model's own
    output, and bf16 rounding would swamp the comparison.
    """
    torch.manual_seed(seed)
    config = tiny_v4_config(**config_overrides)
    hf_model = DeepseekV4ForCausalLM(config).to(torch.float32).eval()
    lens_model = from_hf(hf_model, _ByteTokenizer())
    assert isinstance(lens_model, DeepseekV4LensModel)  # auto-detection sanity
    return lens_model
