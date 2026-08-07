"""The paper's two appendix robustness variants. Usage: variant_fit.py present|frozen

The paper reports its qualitative results are robust to (a) computing only
present-token effects rather than present-plus-future, and (b) freezing
attention patterns. Neither has been tested here, so neither claim has been
checked on this architecture.

--- present ---------------------------------------------------------------
The estimator currently places the cotangent at *every* valid target position
at once, so by causality the gradient at source position p is

    sum_{t' >= p}  d h_t' / d h_p

i.e. present *and* future. Isolating the present term d h_p / d h_p is not free:
a cotangent at a single target position gives one column, so recovering the
diagonal for all p would need one backward per position — about 111x the cost,
which is infeasible (2.7 h per prompt).

What is done instead: **one source position per prompt**. The cotangent is
placed at a single randomly chosen valid position t*, and the gradient is read
only at t*, which gives exactly d h_t*/d h_t*. The backward count is unchanged,
so the fit costs the same as the baseline. Across 100 prompts this samples 100
different positions rather than averaging ~111 positions within each prompt.

That is a real deviation from the baseline's position-averaging and it is
stated rather than buried: it changes the *sampling* of positions, not the
present-vs-future question the variant exists to test.

--- frozen ----------------------------------------------------------------
`eager_attention_forward` is monkeypatched to detach the attention
probabilities before they multiply V, so gradients flow through values but not
through the QK softmax. "eager" is not in ALL_ATTENTION_FUNCTIONS, so dispatch
falls back to the module-level function and the patch takes effect. Everything
else — corpus, band, positions, normalisation — matches the baseline exactly.
"""
# NOTE: paths under /data3 (model weights, fitted lenses in /data3/fan-test/jlens_out)
# are machine-specific to the original 8xB200 host; adjust for your environment.

import math
import sys
import time

import torch

VARIANT = sys.argv[1] if len(sys.argv) > 1 else "present"
assert VARIANT in ("present", "frozen", "onepos"), VARIANT

MODEL = "/data3/fan-test/models/DeepSeek-V4-Flash-0731"
BAND = list(range(19, 40))
DIM_BATCH = 16
MAX_SEQ = 128
SKIP = 16
N_PROMPTS = int(sys.argv[2]) if len(sys.argv) > 2 else 100
OUT = (f"/data3/fan-test/jlens_out/v4flash_variant_{VARIANT}_"
       f"n{N_PROMPTS}_s128.pt")   # keeps n=100 and n=1000 side by side
SEED = 0


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


from transformers import AutoModelForCausalLM, AutoTokenizer, FineGrainedFP8Config
from transformers.models.deepseek_v4 import modeling_deepseek_v4 as M

if VARIANT == "frozen":
    import torch.nn.functional as F
    from torch import nn as _nn
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import repeat_kv

    def frozen_attention_forward(module, query, key, value, attention_mask,
                                 scaling, dropout=0.0, **kwargs):
        key_states = repeat_kv(key, module.num_key_value_groups)
        value_states = repeat_kv(value, module.num_key_value_groups)
        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        sinks = module.sinks.reshape(1, -1, 1, 1).expand(
            query.shape[0], -1, query.shape[-2], -1)
        combined = torch.cat([attn_weights, sinks], dim=-1)
        combined = combined - combined.max(dim=-1, keepdim=True).values
        probs = F.softmax(combined, dim=-1, dtype=combined.dtype)
        scores = probs[..., :-1]
        # THE VARIANT: detach the pattern so no gradient flows through QK/softmax
        scores = scores.detach()
        w = _nn.functional.dropout(scores, p=dropout,
                                   training=module.training).to(value_states.dtype)
        out = torch.matmul(w, value_states).transpose(1, 2).contiguous()
        return out, w

    M.eager_attention_forward = frozen_attention_forward
    log("patched eager_attention_forward: attention patterns FROZEN")

n_gpu = torch.cuda.device_count()
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype="auto", device_map="auto",
    max_memory={i: "146GiB" for i in range(n_gpu)},
    attn_implementation="eager", experts_implementation="grouped_mm",
    quantization_config=FineGrainedFP8Config(dequantize=True))
model.eval()
tok = AutoTokenizer.from_pretrained(MODEL)
log(f"LOADED (variant={VARIANT})")

import jlens.examples as ex
from jlens.fitting import valid_position_mask
from jlens.hf import from_hf
from jlens.hooks import ActivationRecorder
from jlens.lens import JacobianLens
from jlens.protocol import d_source_of, d_target_of, target_module_of

lm = from_hf(model, tok)
D_SRC, D_TGT = d_source_of(lm), d_target_of(lm)
TARGET = lm.n_layers
extra = {TARGET: target_module_of(lm)}
n_passes = math.ceil(D_TGT / DIM_BATCH)
log(f"d_source={D_SRC} d_target={D_TGT} band L{BAND[0]}-{BAND[-1]} "
    f"{n_passes} backwards/prompt")

prompts = ex.load_wikitext_prompts(N_PROMPTS, min_chars=600)
g = torch.Generator().manual_seed(SEED)

acc = {l: torch.zeros(D_TGT, D_SRC, dtype=torch.float32) for l in BAND}
n_done = 0
t0 = time.time()

for _pi, prompt in enumerate(prompts):
    ids = lm.encode(prompt, max_length=MAX_SEQ)
    S = ids.shape[1]
    mask = valid_position_mask(S, skip_first=SKIP)
    valid = mask.nonzero(as_tuple=True)[0]
    if valid.numel() == 0:
        continue

    with ActivationRecorder(lm.layers, at=BAND + [TARGET], start_graph_at=min(BAND),
                            extra_modules=extra) as rec:
        rep = ids.expand(DIM_BATCH, -1)
        lm.forward(rep)
        src_acts = [rec.activations[l] for l in BAND]
        tgt_act = rec.activations[TARGET]

        dev = tgt_act.device
        vpos = valid.to(dev)
        if VARIANT == "present":
            # single position: cotangent there, gradient read only there
            pick = vpos[torch.randint(len(vpos), (1,), generator=g).item()]
            cot_pos = pick.reshape(1)
            read_pos = pick.reshape(1)
        elif VARIANT == "onepos":
            # CONTROL for `present`. Cotangent at ALL positions (so the gradient
            # keeps its future terms, exactly like the baseline) but read at ONE
            # position (exactly like `present`). This isolates the two things
            # `present` changes at once: dropping future effects, and going from
            # ~111 source positions per prompt to 1. present-vs-onepos is then
            # the pure present/future effect, with position sampling held fixed.
            pick = vpos[torch.randint(len(vpos), (1,), generator=g).item()]
            cot_pos = vpos
            read_pos = pick.reshape(1)
        else:
            cot_pos = vpos
            read_pos = vpos

        b_idx = torch.arange(DIM_BATCH, device=dev)
        cot = torch.zeros_like(tgt_act)
        per_prompt = {l: torch.zeros(D_TGT, D_SRC, dtype=torch.float32) for l in BAND}

        for pass_idx, d0 in enumerate(range(0, D_TGT, DIM_BATCH)):
            nd = min(DIM_BATCH, D_TGT - d0)
            cot.zero_()
            cot[b_idx[:nd, None], cot_pos[None, :], d0 + b_idx[:nd, None]] = 1.0
            grads = torch.autograd.grad(
                outputs=tgt_act, inputs=src_acts, grad_outputs=cot,
                retain_graph=(pass_idx < n_passes - 1))
            for l, grad in zip(BAND, grads, strict=True):
                rp = read_pos.to(grad.device, non_blocking=True)
                sel = grad[:nd, rp].float()
                per_prompt[l][d0:d0 + nd, :] = sel.flatten(start_dim=2).mean(1).cpu()

    for l in BAND:                       # same per-prompt normalisation as baseline
        nrm = per_prompt[l].norm()
        if nrm > 0:
            per_prompt[l] /= nrm
        acc[l] += per_prompt[l]
    n_done += 1
    if n_done % 10 == 0:
        el = time.time() - t0
        log(f"  prompt {n_done}/{len(prompts)}  {el/n_done:.0f}s/prompt  "
            f"eta {(len(prompts)-n_done)*el/n_done/3600:.1f}h")

jac = {l: acc[l] / max(n_done, 1) for l in BAND}
lens = JacobianLens(jacobians=jac, n_prompts=n_done, d_model=D_TGT, d_source=D_SRC)
lens.save(OUT)
log(f"FIT DONE variant={VARIANT}, {n_done} prompts, "
    f"{(time.time()-t0)/3600:.2f}h -> {OUT}")
log("DONE")
