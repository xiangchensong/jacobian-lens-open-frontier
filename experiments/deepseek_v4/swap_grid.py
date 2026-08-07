"""Experiment 3.3 -- the swap mode x alpha grid: three swap formulations at five
strengths on the 90 multi-hop items, each scored all-items and conditional on
baseline correctness, splitting failures into target / unchanged / other.

This is where the reported base-conditional swap rate of 0.434 comes from, and
where the target/unchanged/other split exposes the 128x steering-scale error
(alpha=8 landing 0 unchanged and 53 other). The selected mode is written to
`results/exp33_grid.json`, which broadcast.py reads.

The first pass used mode="transfer", whose edit is proportional to the source
coordinate <h, d_from>. It saturated: alpha=2 and alpha=4 gave identical 23/53
and alpha=8 destroyed the model (52/53 "other"). 14 baseline-correct items never
moved at any strength -- the signature of an edit that cannot bite when the
source coordinate is small, since the whole term scales with it.

Two fixes now in jlens.intervene, tested here against the original:
  clamp       -- zero the source coordinate and set the target to a fixed level
                 (the layer's mean residual norm), so the edit no longer depends
                 on how strongly the source happened to be present
  contrastive -- push along normalize(J.T @ (W_U[to] - W_U[from])), the single
                 direction that trades one logit for the other, leaving the
                 orthogonal complement alone
plus vocabulary-mean centring of W_U rows in lens_directions, which removes the
generic "some token here" component every direction otherwise shares.

Scored conditional on baseline correctness, which is the comparison the paper is
making (their models are near ceiling on this task; V4-Flash is at 0.589).
"""

import json
import os
import time

import torch

MODEL = "/data3/fan-test/models/DeepSeek-V4-Flash-0731"
LENS = "/data3/fan-test/jlens_out/v4flash_runAnorm_n100_s128.pt"
BAND = list(range(19, 40))
DEV = "cuda:0"
MODES = ("transfer", "clamp", "contrastive")
ALPHAS = (0.5, 1.0, 2.0, 4.0, 8.0)
SCRATCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(SCRATCH, exist_ok=True)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


from transformers import AutoModelForCausalLM, AutoTokenizer, FineGrainedFP8Config

n_gpu = torch.cuda.device_count()
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype="auto", device_map="auto",
    max_memory={i: "146GiB" for i in range(n_gpu)},
    attn_implementation="eager", experts_implementation="grouped_mm",
    quantization_config=FineGrainedFP8Config(dequantize=True))
model.eval()
tok = AutoTokenizer.from_pretrained(MODEL)
log(f"LOADED in {time.time()-t0:.0f}s")

from jlens.hf import from_hf
from jlens.intervene import swap
from jlens.lens import JacobianLens

lm = from_hf(model, tok)
lens = JacobianLens.load(LENS)
lens.jacobians = {l: lens.jacobians[l].to(DEV) for l in BAND}
lens.source_layers = BAND

items = json.load(open("data/experiments/probe-swap.json"))["items"]


def first_id(w):
    for v in (" " + w, w, " " + w.capitalize(), w.capitalize()):
        e = tok.encode(v, add_special_tokens=False)
        if e:
            return e[0]
    return None


def single_id(w):
    for v in (" " + w, w, " " + w.capitalize(), w.capitalize()):
        e = tok.encode(v, add_special_tokens=False)
        if len(e) == 1:
            return e[0]
    return None


@torch.no_grad()
def greedy(ids, ctx=None):
    if ctx is None:
        return int(model(ids).logits[0, -1].argmax())
    with ctx:
        return int(model(ids).logits[0, -1].argmax())


prepared = []
for it in items:
    src, dst = single_id(it["intermediate"]), single_id(it["swap_to"])
    ans, swp = first_id(it["answer"]), first_id(it["swap_answer"])
    if None in (src, dst, ans, swp):
        continue
    ids = lm.encode(it["prompt"], max_length=256)
    prepared.append((it, ids, src, dst, ans, swp, greedy(ids) == ans))

n_ok = sum(1 for p in prepared if p[-1])
log(f"{len(prepared)} items, baseline correct on {n_ok}")

grid = {}
t0 = time.time()
for mode in MODES:
    for a in ALPHAS:
        c = {"t_all": 0, "t_ok": 0, "u_ok": 0, "o_ok": 0}
        for _it, ids, src, dst, ans, swp, ok in prepared:
            g = greedy(ids, swap(lens, lm, layers=BAND, from_token=src,
                                 to_token=dst, alpha=a, mode=mode))
            hit = g == swp
            c["t_all"] += hit
            if ok:
                c["t_ok"] += hit
                c["u_ok"] += (g == ans)
                c["o_ok"] += not hit and g != ans
        grid[(mode, a)] = c
        log(f"  {mode:<12} a={a:<4} all {c['t_all']}/{len(prepared)}={c['t_all']/len(prepared):.3f}"
            f"   base-ok {c['t_ok']}/{n_ok}={c['t_ok']/max(n_ok,1):.3f}"
            f"   (unchg {c['u_ok']}, other {c['o_ok']})   [{time.time()-t0:.0f}s]")

log("=" * 70)
best = max(grid.items(), key=lambda kv: kv[1]["t_ok"])
log(f"BEST: mode={best[0][0]} alpha={best[0][1]}  "
    f"base-ok {best[1]['t_ok']}/{n_ok} = {best[1]['t_ok']/max(n_ok,1):.3f}")
log("paper: 0.54-0.70 (Haiku 4.5 0.54, Sonnet 4.5 0.70, Opus 4.5 0.70)")
with open(f"{SCRATCH}/exp33_grid.json", "w") as f:
    json.dump({f"{m}|{a}": v for (m, a), v in grid.items()}
              | {"n_items": len(prepared), "n_base_ok": n_ok}, f, indent=1)
log("DONE")
