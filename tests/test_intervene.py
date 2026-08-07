# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Gates for the causal-intervention primitives.

Every causal result downstream rests on these edits doing what they claim, so
each primitive is checked against a property that would break loudly if the
mHC ``[hc_mult, d_model]`` reshape or the direction algebra were wrong:

* the identity gate — a zero-strength edit must be *bit*-identical, not close;
* the direction gate — swapping A for B must move B up and A down, and do so
  monotonically in strength;
* the subspace gate — ablation must null exactly the intended subspace and
  leave its orthogonal complement untouched.

Run against both the multi-stream tiny V4 and the single-stream tiny decoder,
because the reshape path is the one place the two genuinely differ.
"""

from __future__ import annotations

import pytest
import torch

from jlens.fitting import fit
from jlens.intervene import ablate, lens_directions, steer, swap
from jlens.lens import JacobianLens
from tests.tiny import TinyDecoder
from tests.tiny_v4 import tiny_v4_lens_model

PROMPTS = ["the quick brown fox jumps", "a second short prompt here"]


@pytest.fixture(scope="module")
def v4():
    """Tiny multi-stream model plus a genuinely fitted lens."""
    model = tiny_v4_lens_model(seed=0)
    lens = fit(model, PROMPTS, dim_batch=8, max_seq_len=16, skip_first=0)
    return model, lens


@pytest.fixture(scope="module")
def square():
    """Single-stream model plus a fitted lens, to guard the degenerate path."""
    model = TinyDecoder(seed=0)
    lens = fit(model, PROMPTS, dim_batch=8, max_seq_len=16, skip_first=0)
    return model, lens


def _logits(model, lens, ids, ctx=None):
    """Logits at the final position, read off the last fitted layer.

    The whole ``[seq, ...]`` activation is unembedded and the last position
    taken afterwards, rather than slicing the position out first: a bare
    ``[hc_mult, d_model]`` tensor is indistinguishable from ``[batch, d_model]``,
    so ``unembed`` cannot tell whether to collapse the streams. Keeping the
    position axis removes the ambiguity, and works unchanged on the
    single-stream model.
    """
    layers = lens.source_layers
    from jlens.hooks import ActivationRecorder

    rec = ActivationRecorder(model.layers, at=[layers[-1]])
    if ctx is None:
        with torch.no_grad(), rec:
            model.forward(ids)
            act = rec.activations[layers[-1]][0]
    else:
        with ctx, torch.no_grad(), rec:
            model.forward(ids)
            act = rec.activations[layers[-1]][0]
    return model.unembed(act).float()[-1]


def _rank(logits, token):
    return int((logits > logits[token]).sum())


# --------------------------------------------------------------- identity gate
@pytest.mark.parametrize("fixture", ["v4", "square"])
def test_zero_strength_is_bit_identical(fixture, request):
    """alpha=0 / strength=0 must be exactly the identity, not merely close.

    This is the control condition in the introspection protocol, so any drift
    here would show up as a spurious baseline effect in every steering result.
    """
    model, lens = request.getfixturevalue(fixture)
    ids = model.encode(PROMPTS[0], max_length=16)
    base = _logits(model, lens, ids)

    ctxs = [
        steer(lens, model, layers=lens.source_layers, token=1, strength=0.0)
    ] + [
        swap(lens, model, layers=lens.source_layers, from_token=1, to_token=2,
             alpha=0.0, mode=m)
        for m in ("transfer", "clamp", "contrastive")
    ]
    for ctx in ctxs:
        got = _logits(model, lens, ids, ctx)
        assert torch.equal(got, base)


def test_intervention_removes_its_hooks(v4):
    """Leaked hooks would silently contaminate every later measurement."""
    model, lens = v4
    ids = model.encode(PROMPTS[0], max_length=16)
    base = _logits(model, lens, ids)
    with swap(lens, model, layers=lens.source_layers, from_token=1, to_token=2,
              alpha=5.0):
        pass
    assert torch.equal(_logits(model, lens, ids), base)


# -------------------------------------------------------------- direction gate
@pytest.mark.parametrize("fixture", ["v4", "square"])
def test_swap_moves_target_up_and_source_down(fixture, request):
    model, lens = request.getfixturevalue(fixture)
    ids = model.encode(PROMPTS[0], max_length=16)
    base = _logits(model, lens, ids)
    order = base.argsort(descending=True)
    src, dst = int(order[0]), int(order[-1])       # top token -> bottom token

    for mode in ("transfer", "clamp", "contrastive"):
        got = _logits(
            model, lens, ids,
            swap(lens, model, layers=lens.source_layers, from_token=src,
                 to_token=dst, alpha=8.0, mode=mode),
        )
        assert _rank(got, dst) < _rank(base, dst), f"{mode}: target not promoted"
        assert _rank(got, src) > _rank(base, src), f"{mode}: source not demoted"


@pytest.mark.parametrize("fixture", ["v4", "square"])
def test_steer_is_monotone_in_strength(fixture, request):
    """A stronger injection must not make the injected token less likely."""
    model, lens = request.getfixturevalue(fixture)
    ids = model.encode(PROMPTS[0], max_length=16)
    base = _logits(model, lens, ids)
    token = int(base.argsort(descending=True)[-1])

    ranks = [
        _rank(
            _logits(model, lens, ids,
                    steer(lens, model, layers=lens.source_layers, token=token,
                          strength=s)),
            token,
        )
        for s in (0.0, 2.0, 8.0, 32.0)
    ]
    assert ranks == sorted(ranks, reverse=True), f"not monotone: {ranks}"
    assert ranks[-1] < ranks[0], "strong steering had no effect"


# --------------------------------------------------------------- subspace gate
def test_ablate_nulls_exactly_its_subspace(v4):
    """The removed subspace must vanish and its complement must survive."""
    model, lens = v4
    layer = lens.source_layers[-1]
    ids = model.encode(PROMPTS[0], max_length=16)
    tokens = [1, 2, 3]

    from jlens.hooks import ActivationRecorder

    def residual_at(ctx=None):
        rec = ActivationRecorder(model.layers, at=[layer])
        if ctx is None:
            with torch.no_grad(), rec:
                model.forward(ids)
                return rec.activations[layer].detach().reshape(1, ids.shape[1], -1)
        with ctx, torch.no_grad(), rec:
            model.forward(ids)
            return rec.activations[layer].detach().reshape(1, ids.shape[1], -1)

    before = residual_at()
    after = residual_at(
        ablate(lens, model, layers=[layer], token_ids=tokens)
    )

    D = lens_directions(lens, model, layer, tokens).to(before.dtype)
    Q, _ = torch.linalg.qr(D.T.float())
    Q = Q.to(before.dtype)

    # component inside the removed subspace: gone
    assert (after.float() @ Q.float()).abs().max() < 1e-3
    # component orthogonal to it: unchanged
    perp_before = before.float() - (before.float() @ Q.float()) @ Q.float().T
    perp_after = after.float() - (after.float() @ Q.float()) @ Q.float().T
    assert torch.allclose(perp_before, perp_after, atol=1e-4)


def test_ablate_random_control_differs_from_lens_ablation(v4):
    """The matched-norm control must remove a *different* subspace of equal rank."""
    model, lens = v4
    layer = lens.source_layers[-1]
    ids = model.encode(PROMPTS[0], max_length=16)
    gen = torch.Generator().manual_seed(0)

    base = _logits(model, lens, ids)
    real = _logits(model, lens, ids,
                   ablate(lens, model, layers=[layer], token_ids=[1, 2, 3]))
    ctrl = _logits(model, lens, ids,
                   ablate(lens, model, layers=[layer], token_ids=[1, 2, 3],
                          random_control=True, generator=gen))
    assert not torch.equal(real, base)
    assert not torch.equal(real, ctrl)


# ------------------------------------------------------------- position gating
def test_positions_restricts_the_edit(v4):
    """Editing position p must leave strictly-earlier positions untouched.

    Later positions are expected to move — that is attention doing its job —
    so only the causal past is a valid invariant here.
    """
    model, lens = v4
    layer = lens.source_layers[0]
    ids = model.encode(PROMPTS[0], max_length=16)
    p = ids.shape[1] - 2

    from jlens.hooks import ActivationRecorder

    def final_residual(ctx=None):
        last = lens.source_layers[-1]
        rec = ActivationRecorder(model.layers, at=[last])
        if ctx is None:
            with torch.no_grad(), rec:
                model.forward(ids)
                return rec.activations[last].detach().clone()
        with ctx, torch.no_grad(), rec:
            model.forward(ids)
            return rec.activations[last].detach().clone()

    base = final_residual()
    got = final_residual(
        swap(lens, model, layers=[layer], from_token=1, to_token=2, alpha=8.0,
             positions=[p])
    )
    assert torch.allclose(base[0, :p], got[0, :p], atol=1e-5)
    assert not torch.allclose(base[0, p], got[0, p], atol=1e-5)


def test_lens_directions_are_unit_norm(v4):
    model, lens = v4
    D = lens_directions(lens, model, lens.source_layers[-1], [1, 2, 3, 4])
    norms = D.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    assert D.shape == (4, lens.d_source)


def test_unknown_swap_mode_raises(v4):
    model, lens = v4
    with pytest.raises(ValueError, match="unknown swap mode"):
        swap(lens, model, layers=lens.source_layers, from_token=1, to_token=2,
             mode="nonsense")


def test_unknown_layer_raises(v4):
    model, lens = v4
    with pytest.raises(KeyError):
        lens_directions(lens, model, 999, [1])


def test_lens_without_unembed_matrix_raises():
    """A model with no reachable unembedding matrix must fail loudly."""

    class Bare:
        d_model = 4
        n_layers = 1
        layers: list = []

    lens = JacobianLens({0: torch.eye(4)}, n_prompts=1, d_model=4)
    with pytest.raises(AttributeError, match="unembedding matrix"):
        lens_directions(lens, Bare(), 0, [0])
