"""Experiment 5.3 -- the ablation selectivity table: position-adaptive top-k lens
ablation over four layer sub-ranges, each with a matched-rank random control,
scored on both multi-hop reasoning and pretraining next-token agreement.

This is the script behind the headline 5.3 numbers -- multi-hop 0.589 -> 0.100
with the random control at 0.589, pretraining top-1 agreement 0.508 under
J-ablation against 0.922 under the control -- and behind the "guarded" condition
that tests whether the agreement loss is an artifact of the ablated set
overlapping the model's own prediction (0.527, i.e. it is not).

Result to explain: ablating the top-10 lens directions per position across
L19-39 collapses multi-hop reasoning (0.589 -> 0.100, random control 0.589) but
ALSO drops pretraining next-token agreement to 0.508, where the paper reports
automatic prediction staying above 0.90.

Hypothesis: at late band layers the lens is already predicting the next token,
so "ablate the top-10 lens directions here" partly means "ablate the model's own
prediction". 6.2 measured exactly this and the numbers support it -- lens top-10
accuracy rises from 0.021 at L19 to 0.388 at L39, so by the end of the band
roughly two in five positions have the true next token inside the ablated set.
That would make the damage to next-token prediction an artifact of *where* the
ablation is applied, not evidence that J-space carries automatic prediction.

Four conditions separate it:

  full   L19-39   the original -- both effects present
  early  L19-30   lens top-10 accuracy still low here; if the hypothesis holds
                  this should preserve next-token agreement while still
                  collapsing multi-hop, which is the paper's actual claim
  late   L31-39   where the lens has converged on the prediction; should damage
                  next-token agreement most
  guarded L19-39  full band, but the model's own current top-1 prediction is
                  excluded from the ablated set at every position -- a direct
                  test, since it removes workspace content while leaving the
                  prediction alone

Each with a matched-rank random control, because that is what turned the first
Phase D null into a real result.
"""

import json
import os
import time

import torch

MODEL = "/data3/fan-test/models/DeepSeek-V4-Flash-0731"
LENS = "/data3/fan-test/jlens_out/v4flash_runAnorm_n100_s128.pt"
FULL = list(range(19, 40))
EARLY = list(range(19, 31))
LATE = list(range(31, 40))
DEV = "cuda:0"
K = 10
N_PRETRAIN = 15
SCRATCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(SCRATCH, exist_ok=True)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


from transformers import AutoModelForCausalLM, AutoTokenizer, FineGrainedFP8Config

n_gpu = torch.cuda.device_count()
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype="auto", device_map="auto",
    max_memory={i: "146GiB" for i in range(n_gpu)},
    attn_implementation="eager", experts_implementation="grouped_mm",
    quantization_config=FineGrainedFP8Config(dequantize=True))
model.eval()
tok = AutoTokenizer.from_pretrained(MODEL)
log("LOADED")

import jlens.examples as ex
from jlens.hf import from_hf
from jlens.intervene import Intervention
from jlens.lens import JacobianLens

lm = from_hf(model, tok)
lens = JacobianLens.load(LENS)
J = {l: lens.jacobians[l].to(DEV) for l in FULL}
W = model.lm_head.weight
Wm = W.mean(0)


def ablate(layers, k=K, random_control=False, guard=False, seed=0):
    """Position-adaptive top-k ablation.

    guard=True drops the model's own current top-1 prediction from the ablated
    set at each position, so workspace content is removed but the prediction is
    left intact.
    """
    gen = torch.Generator(device=DEV).manual_seed(seed)

    def edit(flat, layer):
        Jl = J[layer]
        x = flat.to(DEV, torch.float32)
        B, S, _ = x.shape
        lg = lm.unembed(x @ Jl.T).float()
        if guard:
            top = lg.topk(k + 1, dim=-1).indices          # one spare
            keep = top[..., :1]                            # the lens's own top-1
            top = torch.where(top == keep, top[..., -1:].expand_as(top), top)
            top = top[..., 1:k + 1]
        else:
            top = lg.topk(k, dim=-1).indices
        rows = W[top.reshape(-1)].to(DEV, torch.float32) - Wm.to(DEV, torch.float32)
        D = (rows @ Jl).reshape(B, S, top.shape[-1], -1)
        if random_control:
            D = torch.randn(D.shape, generator=gen, device=DEV)
        D = D / D.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        Q, _ = torch.linalg.qr(D.transpose(-1, -2))
        c = torch.einsum("bsd,bsdr->bsr", x, Q)
        return (x - torch.einsum("bsr,bsdr->bsd", c, Q)).to(flat.dtype).to(flat.device)

    return Intervention(lm, layers, edit)


CONDS = {
    "full L19-39":       lambda: ablate(FULL),
    "full L19-39 rand":  lambda: ablate(FULL, random_control=True),
    "early L19-30":      lambda: ablate(EARLY),
    "early L19-30 rand": lambda: ablate(EARLY, random_control=True),
    "late L31-39":       lambda: ablate(LATE),
    "guarded L19-39":    lambda: ablate(FULL, guard=True),
}


@torch.no_grad()
def run():
    prompts = ex.load_wikitext_prompts(N_PRETRAIN, min_chars=600)
    base = []
    for p in prompts:
        ids = lm.encode(p, max_length=128)
        base.append((ids, model(ids).logits[0].argmax(-1).cpu()))

    items = json.load(open("data/experiments/probe-swap.json"))["items"]
    prep = []
    for it in items:
        a = tok.encode(" " + it["answer"], add_special_tokens=False)
        if a:
            prep.append((lm.encode(it["prompt"], max_length=256), a[0]))
    unabl = sum(1 for ids, a in prep
                if int(model(ids).logits[0, -1].argmax()) == a)
    log(f"unablated multi-hop {unabl}/{len(prep)} = {unabl/len(prep):.3f}")

    log(f"  {'condition':<20} {'next-token agree':>17} {'multi-hop':>11}")
    log(f"  {'unablated':<20} {1.0:>17.3f} {unabl/len(prep):>11.3f}")
    out = {}
    for name, mk in CONDS.items():
        ctx = mk()
        ag = tot = 0
        for ids, bp in base:
            with ctx:
                pr = model(ids).logits[0].argmax(-1).cpu()
            ag += int((pr == bp).sum())
            tot += bp.numel()
        ok = 0
        for ids, a in prep:
            with ctx:
                ok += int(model(ids).logits[0, -1].argmax()) == a
        out[name] = {"agree": ag / tot, "multihop": ok / len(prep)}
        log(f"  {name:<20} {ag/tot:>17.3f} {ok/len(prep):>11.3f}")
    json.dump(out, open(f"{SCRATCH}/exp53_selectivity.json", "w"), indent=1)
    log("  hypothesis holds if 'early' keeps agreement high AND multi-hop low;")
    log("  it also holds if 'guarded' recovers agreement relative to 'full'.")


try:
    run()
except Exception as exc:
    import traceback
    log(f"!! FAILED {type(exc).__name__}: {exc}")
    traceback.print_exc()
log("DONE")
