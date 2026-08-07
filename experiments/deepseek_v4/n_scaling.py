"""The n-scaling curve: re-run the 3.3 multi-hop swap against lenses fitted on
n=20, n=60 and n=100 prompts, all else held fixed, to measure whether swap
success is still improving with corpus size.

This is the pre-registration evidence for the n=1000 decision -- the curve is
flat from n=60 (0.434 at n=60 and at n=100) -- which compare_n100_n1000.py then
tested against the full-corpus fit.

3.3 (0.434 vs 0.54-0.70), 4.1 (0.208 vs 0.526) and 5.1 (0.286 vs ~1.00) all fall
short, and they have one thing in common: every one of them depends on the
*quality of a lens direction*, not on the readout. The obvious systematic
suspect is that my lens is averaged over 100 prompts where the paper uses 1000.

Rather than assume that and spend 26 GPU-hours fitting n=1000, measure the
curve first with the lenses already on disk (n=20, n=60, n=100). If swap
success is still climbing at n=100, more prompts will help and the fit is worth
launching. If it has plateaued, n is not the bottleneck and the shortfall is
about the model, not the estimator.
"""
import json
import os
import time

import torch

MODEL = "/data3/fan-test/models/DeepSeek-V4-Flash-0731"
SCRATCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(SCRATCH, exist_ok=True)
LENSES = [
 ("n=20",  "/data3/fan-test/jlens_out/v4flash_lens_n20_s128.pt"),
 ("n=60",  "/data3/fan-test/jlens_out/v4flash_runAnorm_INTERIM_n60_s128.pt"),
 ("n=100", "/data3/fan-test/jlens_out/v4flash_runAnorm_n100_s128.pt"),
]
BAND = list(range(19,40))
DEV="cuda:0"
ALPHA=2.0


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
from jlens.hf import from_hf
from jlens.intervene import swap
from jlens.lens import JacobianLens

lm = from_hf(model, tok)
items = json.load(open("data/experiments/probe-swap.json"))["items"]


def first_id(w):
    for v in (" "+w, w, " "+w.capitalize(), w.capitalize()):
        e = tok.encode(v, add_special_tokens=False)
        if e:
            return e[0]


def single_id(w):
    for v in (" "+w, w, " "+w.capitalize(), w.capitalize()):
        e = tok.encode(v, add_special_tokens=False)
        if len(e)==1:
            return e[0]


@torch.no_grad()
def greedy(ids, ctx=None):
    if ctx is None:
        return int(model(ids).logits[0,-1].argmax())
    with ctx:
        return int(model(ids).logits[0,-1].argmax())


prep=[]
for it in items:
    s,d = single_id(it["intermediate"]), single_id(it["swap_to"])
    a,w = first_id(it["answer"]), first_id(it["swap_answer"])
    if None in (s,d,a,w):
        continue
    ids = lm.encode(it["prompt"], max_length=256)
    prep.append((ids,s,d,a,w, greedy(ids)==a))
n_ok = sum(1 for p in prep if p[-1])
log(f"{len(prep)} items, baseline correct {n_ok}")
log(f"  {'lens':<8} {'layers':<10} {'all':>12} {'base-ok':>12}")
out={}
for tag, path in LENSES:
    lens = JacobianLens.load(path)
    have = [l for l in BAND if l in lens.jacobians]
    if len(have) < len(BAND):
        log(f"  {tag}: only {len(have)}/{len(BAND)} band layers present; using those")
    lens.jacobians = {l: lens.jacobians[l].to(DEV) for l in have}
    lens.source_layers = have
    t=tok_ok=0
    for ids,s,d,_a,w,ok in prep:
        g = greedy(ids, swap(lens, lm, layers=have, from_token=s, to_token=d,
                             alpha=ALPHA, mode="transfer"))
        t += (g==w)
        tok_ok += (ok and g==w)
    out[tag]={"all":t/len(prep),"base_ok":tok_ok/max(n_ok,1),"layers":len(have)}
    log(f"  {tag:<8} L{have[0]}-{have[-1]:<7} {t}/{len(prep)}={t/len(prep):.3f}"
        f"   {tok_ok}/{n_ok}={tok_ok/max(n_ok,1):.3f}")
    del lens
    torch.cuda.empty_cache()
json.dump(out, open(f"{SCRATCH}/exp_nscaling.json","w"), indent=1)
log("  climbing -> fit n=1000 is worth ~26 GPU-hours; flat -> n is not the bottleneck")
log("DONE")
