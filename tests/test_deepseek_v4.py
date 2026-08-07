# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Rectangular-Jacobian gate for the multi-stream (mHC) path.

DeepSeek-V4 keeps ``hc_mult`` parallel residual streams, so ``J_l`` becomes
``[d_target, d_source] = [hidden, hc_mult * hidden]`` and the Jacobian target
moves out of the block stack onto ``hc_head``. These tests run the real
modeling code (see ``tests/tiny_v4.py``) and check the four things that can go
wrong silently: the shapes, the gradient itself (against finite differences),
the column order of the flattened source, and the ``unembed`` dispatch that
makes the final-layer readout reproduce the model's own logits.
"""

from __future__ import annotations

import pytest
import torch

from jlens.fitting import fit, jacobian_for_prompt, valid_position_mask
from jlens.lens import JacobianLens
from jlens.vis import compute_slice

from .tiny_v4 import HC_MULT, HIDDEN_SIZE, N_LAYERS, tiny_v4_lens_model

#: Long enough to clear ``SKIP_FIRST_N_POSITIONS`` with positions to spare.
PROMPT = "the quick brown fox jumps over the lazy dog " * 2
SOURCE_LAYERS = [0, 1, 2]
D_SOURCE = HC_MULT * HIDDEN_SIZE


def test_model_exposes_rectangular_widths():
    model = tiny_v4_lens_model()
    assert (model.n_layers, model.d_model) == (N_LAYERS, HIDDEN_SIZE)
    assert (model.d_source, model.d_target) == (D_SOURCE, HIDDEN_SIZE)
    assert model.target_module is model._hf_model.model.hc_head
    # A block really does emit hc_mult streams -- the premise of all of this.
    with torch.no_grad():
        out = model._hf_model.model.layers[0](
            torch.zeros(1, 4, HC_MULT, HIDDEN_SIZE),
            position_embeddings={
                k: (torch.zeros(1, 4, 4), torch.ones(1, 4, 4))
                for k in ("main", "compress")
            },
            position_ids=torch.arange(4)[None],
            attention_mask=None,
            input_ids=torch.zeros(1, 4, dtype=torch.long),
        )
    assert out.shape == (1, 4, HC_MULT, HIDDEN_SIZE)


def test_jacobian_shapes_and_column_order():
    """Gate 1: ``J_l`` is ``[d_target, d_source]``, and its columns are the
    source streams flattened row-major -- the same order
    :meth:`JacobianLens.transport` uses."""
    model = tiny_v4_lens_model()
    position_grads: dict[int, torch.Tensor] = {}
    jacobians, seq_len, n_valid = jacobian_for_prompt(
        model,
        PROMPT,
        SOURCE_LAYERS,
        dim_batch=8,
        max_seq_len=64,
        position_grads=position_grads,
    )
    assert set(jacobians) == set(SOURCE_LAYERS)
    for J in jacobians.values():
        assert J.shape == (HIDDEN_SIZE, D_SOURCE) and J.dtype == torch.float32
    assert n_valid > 0 and seq_len > n_valid

    # The debug grads are the same numbers before the position mean, laid out
    # [d_target, seq_len, hc_mult, d]. Reducing them by hand must reproduce J,
    # which pins column (k, i) of J to k * d + i.
    valid = valid_position_mask(seq_len).nonzero(as_tuple=True)[0]
    for layer in SOURCE_LAYERS:
        grads = position_grads[layer]
        assert grads.shape == (HIDDEN_SIZE, seq_len, HC_MULT, HIDDEN_SIZE)
        by_hand = grads[:, valid].mean(dim=1).flatten(start_dim=1)
        torch.testing.assert_close(by_hand, jacobians[layer], rtol=0, atol=1e-6)


def test_jacobian_matches_finite_differences():
    """Gate 2: the estimator against central differences of the model itself.

    For a source entry ``X_l[p, k, i]`` and a target dim ``j``, the recorded
    gradient must equal ``d/dX_l[p, k, i] sum_{p' valid} hc_head(X_L)[p', j]``.
    This is the one check that would catch a wrong cotangent, a wrong flatten,
    or a graph rooted in the wrong place -- all of which produce plausible
    numbers rather than an exception. It also pins the estimator's batch
    replication: the analytic side comes from a ``dim_batch``-wide forward, the
    finite-difference side from a batch of one, so any batch-dependent
    behaviour in the model would show up here as a mismatch.

    The difference is the 4th-order (Richardson) central formula rather than the
    plain 3-point one. In fp32 the plain formula's error bottoms out around
    ``2e-5`` absolute -- truncation ``O(eps^2)`` above the optimum ``eps``,
    roundoff ``~eps_fp32 * |f| / eps`` below it -- which is the same order as a
    typical gradient entry here, so it cannot certify a relative tolerance. The
    4th-order formula pushes truncation out of the way and drops the floor to
    ``~8e-6``, comfortably below the entries being compared. (fp64 would be the
    obvious alternative and is not available: the MoE ``grouped_mm`` kernel
    takes only fp32/bf16/fp16.)

    Two assertions, because a randomly drawn entry may be near zero:
    every sample must agree within ``1e-3 * |analytic| + atol`` with ``atol``
    the measured roundoff floor, and every sample whose gradient is clear of
    that floor must agree to a relative ``1e-3`` outright.
    """
    model = tiny_v4_lens_model()
    position_grads: dict[int, torch.Tensor] = {}
    _, seq_len, _ = jacobian_for_prompt(
        model,
        PROMPT,
        SOURCE_LAYERS,
        dim_batch=8,
        max_seq_len=64,
        position_grads=position_grads,
    )
    input_ids = model.encode(PROMPT, max_length=64)
    valid = valid_position_mask(seq_len).nonzero(as_tuple=True)[0]

    def summed_target(layer: int, p: int, k: int, i: int, eps: float) -> torch.Tensor:
        """``sum_{p' valid} hc_head(X_L)[p']`` with ``X_l[p, k, i] += eps``."""

        def perturb(module, inputs, output):
            perturbed = output.clone()
            perturbed[0, p, k, i] += eps
            return perturbed

        captured: dict[str, torch.Tensor] = {}
        handles = [
            model.layers[layer].register_forward_hook(perturb),
            model.target_module.register_forward_hook(
                lambda module, inputs, output: captured.__setitem__("out", output)
            ),
        ]
        try:
            with torch.no_grad():
                model.forward(input_ids)
        finally:
            for handle in handles:
                handle.remove()
        # Sum in fp64: the model is fp32, but the reduction need not add to it.
        return captured["out"][0, valid].double().sum(dim=0)

    eps = 1.2e-2
    atol = 1.5e-5  # ~2x the measured fp32 floor of this difference
    generator = torch.Generator().manual_seed(0)

    def draw(high: int) -> int:
        return int(torch.randint(high, (1,), generator=generator))

    worst_relative = 0.0
    n_above_floor = 0
    for _ in range(24):
        layer = SOURCE_LAYERS[draw(len(SOURCE_LAYERS))]
        p = int(valid[draw(len(valid))])
        k, i, j = draw(HC_MULT), draw(HIDDEN_SIZE), draw(HIDDEN_SIZE)
        near = summed_target(layer, p, k, i, eps) - summed_target(layer, p, k, i, -eps)
        far = summed_target(layer, p, k, i, 2 * eps) - summed_target(
            layer, p, k, i, -2 * eps
        )
        finite_difference = float((8 * near - far)[j] / (12 * eps))
        analytic = float(position_grads[layer][j, p, k, i])
        error = abs(finite_difference - analytic)
        assert error <= 1e-3 * abs(analytic) + atol, (
            f"l={layer} p={p} k={k} i={i} j={j}: "
            f"analytic={analytic:.6e} finite-difference={finite_difference:.6e}"
        )
        if abs(analytic) > 1e-3:  # >100x the floor: relative error is meaningful
            n_above_floor += 1
            worst_relative = max(worst_relative, error / abs(analytic))
    # Most samples must be resolvable, else the atol above is doing all the work.
    assert n_above_floor >= 12
    assert worst_relative < 1e-3, worst_relative


def test_readout_identity():
    """Gate 3: ``apply``'s ``model_logits`` are the model's own logits.

    This is what proves the ``unembed`` dispatch: the final block's activation
    is still ``[seq, hc_mult, d]``, so reproducing the model's logits from it
    requires routing through ``hc_head`` before the norm and LM head.
    """
    model = tiny_v4_lens_model()
    lens = JacobianLens(
        jacobians={l: torch.randn(HIDDEN_SIZE, D_SOURCE) for l in SOURCE_LAYERS},
        n_prompts=1,
        d_model=HIDDEN_SIZE,
        d_source=D_SOURCE,
    )
    lens_logits, model_logits, input_ids = lens.apply(model, PROMPT, max_seq_len=64)
    with torch.no_grad():
        reference = model._hf_model(input_ids=input_ids, use_cache=False).logits[0]
    torch.testing.assert_close(model_logits, reference.float().cpu(), rtol=0, atol=0)

    seq_len, vocab_size = reference.shape
    for logits in lens_logits.values():
        assert logits.shape == (seq_len, vocab_size)
    # The vanilla-logit-lens baseline reads an uncollapsed residual through
    # hc_head instead of J_l, and must agree with the model at the last layer.
    baseline, _, _ = lens.apply(
        model, PROMPT, max_seq_len=64, use_jacobian=False, layers=[N_LAYERS - 1]
    )
    torch.testing.assert_close(baseline[N_LAYERS - 1], model_logits, rtol=0, atol=0)


def test_fit_save_load_apply_slice(tmp_path):
    """Gate 5: the whole pipeline on the tiny V4."""
    model = tiny_v4_lens_model()
    prompts = ["abcdefghij klmnopqrst " * 3, "the quick brown fox jumps " * 2]
    lens = fit(model, prompts, dim_batch=16, max_seq_len=48)
    # Default source layers are every block: the target is the virtual index
    # n_layers (hc_head), so unlike a single-stream model no block is excluded.
    assert lens.source_layers == list(range(N_LAYERS))
    assert lens.n_prompts == 2
    assert (lens.d_model, lens.d_source) == (HIDDEN_SIZE, D_SOURCE)

    path = tmp_path / "lens.pt"
    lens.save(str(path))
    reloaded = JacobianLens.load(str(path))
    assert (reloaded.d_model, reloaded.d_source) == (HIDDEN_SIZE, D_SOURCE)
    for layer in lens.source_layers:
        torch.testing.assert_close(
            reloaded.jacobians[layer], lens.jacobians[layer], rtol=0, atol=2e-3
        )  # fp16 round-trip

    lens_logits, model_logits, input_ids = reloaded.apply(
        model, PROMPT, layers=[0, 2], positions=[-1], max_seq_len=48
    )
    assert model_logits.shape == (1, model._hf_model.config.vocab_size)
    for logits in lens_logits.values():
        assert logits.shape == model_logits.shape

    slice_data = compute_slice(model, reloaded, PROMPT, top_n=3, max_seq_len=48)
    assert slice_data.layers == list(range(N_LAYERS))
    assert slice_data.top_ids.shape == (slice_data.seq_len, N_LAYERS, 3)


def test_block_target_is_rejected_not_miscomputed():
    """A block output is 4-D, and the one-hot cotangent indexes with exactly
    three indices -- so targeting a block would index the wrong axis and return
    a plausible, wrong ``J_l``. It has to raise instead."""
    model = tiny_v4_lens_model()
    with pytest.raises(ValueError, match="three indices"):
        jacobian_for_prompt(
            model, PROMPT, [0], target_layer=N_LAYERS - 1, dim_batch=8, max_seq_len=48
        )
    # The virtual index is one past the last block; anything beyond is still out
    # of range.
    with pytest.raises(ValueError, match="out of range"):
        jacobian_for_prompt(
            model, PROMPT, [0], target_layer=N_LAYERS + 1, dim_batch=8, max_seq_len=48
        )


def test_transport_accepts_streams_and_flat():
    """``transport`` flattens the stream axis itself, so callers holding a raw
    block activation and callers holding a pre-flattened one agree."""
    lens = JacobianLens(
        jacobians={0: torch.randn(HIDDEN_SIZE, D_SOURCE)},
        n_prompts=1,
        d_model=HIDDEN_SIZE,
        d_source=D_SOURCE,
    )
    streams = torch.randn(5, HC_MULT, HIDDEN_SIZE)
    torch.testing.assert_close(
        lens.transport(streams, 0), lens.transport(streams.flatten(-2), 0)
    )
    assert lens.transport(streams, 0).shape == (5, HIDDEN_SIZE)
