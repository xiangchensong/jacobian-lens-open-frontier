"""Experiment 1.3 -- what share of a concept vector's variance lives in J-space,
plus the position-adaptive form of the 5.3 ablation the measurement motivates.

1.3 is the reported V4 J-space compression figure: the top-25 lens directions
capture 0.4% of a concept vector's variance (paper: 6-15%) while still carrying
the causal effect that 5.3 demonstrates. The concept vector is a category's mean
band residual minus the grand mean over categories, which is the paper's
denominator; measuring against the whole residual instead gives 0.16% and a
misleading null.

5.3 ablation was a null: pretraining agreement 0.905 (J) vs 0.910 (random),
multi-hop 0.600 (J) vs 0.611 (random) against 0.589 unablated. Nothing moved,
and the control was indistinguishable from the treatment -- the signature of an
intervention that is not doing anything at all.

The reason is in 1.3's number: my J-space held 0.16% of residual variance. I
ablated the directions of 10 *globally frequent* tokens, a fixed rank-11
subspace of a 16384-dimensional space (0.07% of it). Removing that cannot
matter. The paper ablates the top-k directions *active at each position*, which
is a different and much larger object: it removes what the workspace is actually
holding right then, and it moves with the content.

So:

5.3 v2  position-adaptive ablation. At every (layer, position), take the k
        tokens the lens ranks highest *there*, build their directions, and
        project them out. Still k=10 per position, but now tracking content.
        Matched-norm random control of the same rank, as before.

1.3 v2  decompose a *concept vector*, not the whole residual. The paper's
        denominator is a concept's own activation difference -- mean residual
        over prompts implying a concept minus the mean over all concepts --
        which is what makes "6-15% of variance, most of the causal effect" a
        meaningful statement. Measuring against the full residual instead is
        how I got 0.16% and a misleading null.
"""

import json
import os
import time
from collections import defaultdict

import torch

MODEL = "/data3/fan-test/models/DeepSeek-V4-Flash-0731"
LENS = "/data3/fan-test/jlens_out/v4flash_runAnorm_n100_s128.pt"
BAND = list(range(19, 40))
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
from jlens.hooks import ActivationRecorder
from jlens.intervene import Intervention
from jlens.lens import JacobianLens

lm = from_hf(model, tok)
lens = JacobianLens.load(LENS)
J = {l: lens.jacobians[l].to(DEV) for l in BAND}
W = model.lm_head.weight
W_MEAN = W.mean(0)


def adaptive_ablate(k=K, random_control=False, seed=0):
    """Project out the k directions the lens ranks highest at each position."""
    gen = torch.Generator(device=DEV).manual_seed(seed)

    def edit(flat, layer):
        Jl = J[layer]
        x = flat.to(DEV, torch.float32)
        B, S, _ = x.shape
        lg = lm.unembed(x @ Jl.T).float()                     # [B,S,vocab]
        top = lg.topk(k, dim=-1).indices                      # [B,S,k]
        rows = (W[top.reshape(-1)].to(DEV, torch.float32)
                - W_MEAN.to(DEV, torch.float32))              # [B*S*k, d_tgt]
        D = (rows @ Jl).reshape(B, S, k, -1)                  # [B,S,k,d_src]
        if random_control:
            D = torch.randn(D.shape, generator=gen, device=DEV)
        D = D / D.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        # orthonormalise the k directions at each position, then project out
        Q, _ = torch.linalg.qr(D.transpose(-1, -2))           # [B,S,d_src,r]
        coef = torch.einsum("bsd,bsdr->bsr", x, Q)
        out = x - torch.einsum("bsr,bsdr->bsd", coef, Q)
        return out.to(flat.dtype).to(flat.device)

    return Intervention(lm, BAND, edit)


# ------------------------------------------------------------------ 5.3 v2
@torch.no_grad()
def exp53():
    log("=" * 70)
    log(f"(5.3 v2) position-adaptive ablation of top-{K} lens directions")
    prompts = ex.load_wikitext_prompts(N_PRETRAIN, min_chars=600)
    conds = {"J-space": adaptive_ablate(),
             "random control": adaptive_ablate(random_control=True)}

    log("  (a) pretraining top-1 agreement (automatic prediction)")
    base = []
    for p in prompts:
        ids = lm.encode(p, max_length=128)
        base.append((ids, model(ids).logits[0].argmax(-1).cpu()))
    for name, ctx in conds.items():
        ag = tot = 0
        for ids, bp in base:
            with ctx:
                pr = model(ids).logits[0].argmax(-1).cpu()
            ag += int((pr == bp).sum())
            tot += bp.numel()
        log(f"      {name:<16} {ag}/{tot} = {ag/tot:.3f}")

    log("  (b) multi-hop greedy accuracy (flexible reasoning)")
    items = json.load(open("data/experiments/probe-swap.json"))["items"]
    prep = []
    for it in items:
        a = tok.encode(" " + it["answer"], add_special_tokens=False)
        if a:
            prep.append((lm.encode(it["prompt"], max_length=256), a[0]))
    ok = sum(1 for ids, a in prep if int(model(ids).logits[0, -1].argmax()) == a)
    log(f"      {'unablated':<16} {ok}/{len(prep)} = {ok/len(prep):.3f}")
    for name, ctx in conds.items():
        ok = 0
        for ids, a in prep:
            with ctx:
                ok += int(model(ids).logits[0, -1].argmax()) == a
        log(f"      {name:<16} {ok}/{len(prep)} = {ok/len(prep):.3f}")
    log("  paper: automatic ~baseline, flexible collapses, random control flat")


# ------------------------------------------------------------------ 1.3 v2
@torch.no_grad()
def exp13(k=25):
    log("=" * 70)
    log(f"(1.3 v2) concept-vector decomposition, k={k} lens directions")
    items = json.load(open("data/experiments/probe-swap.json"))["items"]
    by_cat = defaultdict(list)
    for it in items:
        by_cat[it.get("category", "?")].append(it)
    cats = [c for c, v in by_cat.items() if len(v) >= 3][:6]
    log(f"  {len(cats)} categories with >=3 items")

    # mean residual per category, and the grand mean, per band layer
    sums = {c: {l: None for l in BAND} for c in cats}
    counts = {c: 0 for c in cats}
    for c in cats:
        for it in by_cat[c]:
            ids = lm.encode(it["prompt"], max_length=256)
            with ActivationRecorder(lm.layers, at=BAND) as rec:
                lm.forward(ids)
                for l in BAND:
                    h = rec.activations[l][0, -1].detach().to(DEV).float().flatten()
                    sums[c][l] = h if sums[c][l] is None else sums[c][l] + h
            counts[c] += 1
    grand = {l: sum(sums[c][l] / counts[c] for c in cats) / len(cats) for l in BAND}

    shares = []
    for c in cats:
        for l in BAND:
            v = sums[c][l] / counts[c] - grand[l]          # the concept vector
            lg = lm.unembed(v.unsqueeze(0) @ J[l].T).float().reshape(-1)
            top = lg.topk(k).indices
            rows = W[top].to(DEV, torch.float32) - W_MEAN.to(DEV, torch.float32)
            D = rows @ J[l]
            D = D / D.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            Q, _ = torch.linalg.qr(D.T)
            proj = Q @ (Q.T @ v)
            tot = float(v.pow(2).sum())
            if tot > 0:
                shares.append(float(proj.pow(2).sum()) / tot)
    if shares:
        s = sorted(shares)
        log(f"  J-space share of CONCEPT-VECTOR variance: median "
            f"{s[len(s)//2]:.4f}  mean {sum(s)/len(s):.4f}  (n={len(s)})")
        log("  paper: 6-15% of variance carries most of the causal effect")
        log("  (first pass measured against the whole residual and got 0.0016 --"
            " wrong denominator)")
    json.dump({"shares": shares},
              open(f"{SCRATCH}/exp13_concept_decomp.json", "w"))


for fn in (exp53, exp13):
    try:
        fn()
    except Exception as exc:
        import traceback
        log(f"!! {fn.__name__} FAILED {type(exc).__name__}: {exc}")
        traceback.print_exc()

log("DONE")
