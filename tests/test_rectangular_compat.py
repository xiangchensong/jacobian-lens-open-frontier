# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Backward compatibility of the serialised formats after ``J_l`` went
rectangular.

``d_source`` / ``d_target`` were added to both the lens file and the fit
checkpoint. Artifacts written before that carry neither, and must still load
and resume as the square lenses they are -- the released Qwen lenses on the Hub
are exactly such files.
"""

from __future__ import annotations

import torch

from jlens.fitting import fit
from jlens.lens import JacobianLens

from .tiny import TinyDecoder


def test_lens_file_without_d_source_loads_as_square(tmp_path):
    path = tmp_path / "old_lens.pt"
    torch.save(
        {
            "J": {0: torch.randn(6, 6, dtype=torch.float16)},
            "n_prompts": 3,
            "source_layers": [0],
            "d_model": 6,
        },
        path,
    )
    lens = JacobianLens.load(str(path))
    assert lens.d_model == 6 and lens.d_source == 6
    residual = torch.randn(2, 6)
    torch.testing.assert_close(
        lens.transport(residual, 0), residual @ lens.jacobians[0].T
    )


def test_fit_checkpoint_without_widths_resumes(tmp_path):
    """The compatibility check skips keys a checkpoint predates, so an old
    checkpoint resumes rather than being rejected on a field it never had."""
    model = TinyDecoder(n_layers=4, d_model=8)
    prompts = ["abcdefghij " * 5, "klmnopqrst " * 5]
    checkpoint = str(tmp_path / "ckpt.pt")

    reference = fit(model, prompts, source_layers=[0, 2], dim_batch=4, max_seq_len=64)
    fit(
        model,
        prompts[:1],
        source_layers=[0, 2],
        dim_batch=4,
        max_seq_len=64,
        checkpoint_path=checkpoint,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    del state["d_source"], state["d_target"]  # as an older jlens would have written
    torch.save(state, checkpoint)

    resumed = fit(
        model,
        prompts,
        source_layers=[0, 2],
        dim_batch=4,
        max_seq_len=64,
        checkpoint_path=checkpoint,
    )
    assert resumed.n_prompts == 2
    for layer in [0, 2]:
        torch.testing.assert_close(resumed.jacobians[layer], reference.jacobians[layer])


def test_normalize_per_prompt_equalises_contributions(tmp_path):
    """``normalize_per_prompt`` makes every prompt weigh the same.

    The readout applies a final norm before the LM head, so a uniform rescaling
    of ``J_l`` is invisible in the logits -- which makes the plain mean's
    implicit weighting-by-Jacobian-norm arbitrary. This checks the option does
    what it says, and that leaving it off changes nothing.
    """
    model = TinyDecoder(n_layers=4, d_model=8)
    prompts = ["abcdefghij " * 5, "klmnopqrst " * 5]
    kw = dict(source_layers=[0, 2], dim_batch=4, max_seq_len=64)

    default = fit(model, prompts, **kw)
    explicit_off = fit(model, prompts, normalize_per_prompt=False, **kw)
    for layer in [0, 2]:
        torch.testing.assert_close(
            default.jacobians[layer], explicit_off.jacobians[layer], rtol=0, atol=0
        )

    normalised = fit(model, prompts, normalize_per_prompt=True, **kw)
    # Each prompt contributed a unit-norm J, so the mean of 2 has norm <= 1
    # (equality only if the two prompts' Jacobians were identical).
    for layer in [0, 2]:
        assert normalised.jacobians[layer].norm().item() <= 1.0 + 1e-6
    # Same layers, same shapes, still a usable lens.
    assert normalised.source_layers == [0, 2]
    assert normalised.jacobians[0].shape == default.jacobians[0].shape
