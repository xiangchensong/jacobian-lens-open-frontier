"""The dual-task capacity experiment: hold a concept, an arithmetic problem, or
both, while writing a carrier sentence, and measure lens reachability of each
over the generated span.

This is the source of the arithmetic observation reported in Part II: concepts
reach lens rank <=5 on 92% of trials, while the arithmetic answer never reaches
it -- not even when it is the model's only task -- so the interference contrast
cannot be read on the math side at all.

data/experiments/dual-task.json: the model holds one or two covert tasks while
copying a carrier sentence. A task is "reachable" if any of its target tokens
hits lens rank <=5 anywhere in the band over the response span; interference is
single-task reachability minus dual-task reachability. The paper's claim is that
the workspace has limited capacity, so holding two things at once costs you.
"""
import json
import os
import time

import torch

MODEL="/data3/fan-test/models/DeepSeek-V4-Flash-0731"
LENS="/data3/fan-test/jlens_out/v4flash_band_n1000_s128.pt"
BAND=list(range(19,40))
DEV="cuda:0"
TOPK=5
SCRATCH=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(SCRATCH, exist_ok=True)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


from transformers import AutoModelForCausalLM, AutoTokenizer, FineGrainedFP8Config

n_gpu=torch.cuda.device_count()
model=AutoModelForCausalLM.from_pretrained(MODEL, dtype="auto", device_map="auto",
    max_memory={i:"146GiB" for i in range(n_gpu)}, attn_implementation="eager",
    experts_implementation="grouped_mm",
    quantization_config=FineGrainedFP8Config(dequantize=True))
model.eval()
tok=AutoTokenizer.from_pretrained(MODEL)
log("LOADED")
import sys

sys.path.insert(0, f"{MODEL}/encoding")
import encoding_dsv4 as DSV4

from jlens.hf import from_hf
from jlens.hooks import ActivationRecorder
from jlens.lens import JacobianLens

lm=from_hf(model,tok)
lens=JacobianLens.load(LENS)
have=[l for l in BAND if l in lens.jacobians]
J={l:lens.jacobians[l].to(DEV) for l in have}
d=json.load(open("data/experiments/dual-task.json"))
carrier=d["carrier_sentence"]
pairs={p["key"]:p for p in d["pairs"]}
log(f"{len(pairs)} concept/math pairs, carrier={carrier[:44]!r}")

def ids_of(ws):
    o=set()
    for w in ws:
        for v in (w," "+w,w.capitalize()," "+w.capitalize()):
            e=tok.encode(v,add_special_tokens=False)
            if len(e)==1:
                o.add(e[0])
    return sorted(o)

def chat(user, prefill=""):
    return DSV4.encode_messages([{"role":"user","content":user}],
                                thinking_mode="chat")+prefill

@torch.no_grad()
def reachable(prompt, targets, span):
    if not targets:
        return None
    ids=lm.encode(prompt, max_length=512)
    S=ids.shape[1]
    lo,hi=max(0,span[0]),min(S,span[1])
    if hi<=lo:
        return None
    with ActivationRecorder(lm.layers, at=have) as rec:
        lm.forward(ids)
        acts={l:rec.activations[l][0].detach() for l in have}
    best=None
    for l in have:
        H=acts[l].to(DEV).float()[lo:hi]
        lg=lm.unembed(H.flatten(1)@J[l].T).float()
        r=int((lg>lg[:,targets].max(dim=1,keepdim=True).values).sum(-1).min())
        best=r if best is None else min(best,r)
    return best

def build(instr):
    p=chat(f"{instr}\n\nWrite this sentence: {carrier}", prefill=" "+carrier)
    n_car=len(tok.encode(" "+carrier, add_special_tokens=False))
    ids=lm.encode(p, max_length=512)
    return p,(ids.shape[1]-n_car, ids.shape[1])

res={"concept alone":[0,0],"math alone":[0,0],
     "both (concept scored)":[0,0],"both (math scored)":[0,0]}
t0=time.time()
for k,pr in list(pairs.items())[:12]:
    ctgt=ids_of(pr["concept_words"])
    ans=pr["base"]**pr["exp"]-pr["sub"]
    mtgt=ids_of([str(ans)])
    if not ctgt or not mtgt:
        continue
    ci=f"Concentrate on {pr['concept']} while you write the sentence."
    mi=f"Work out {pr['base']}^{pr['exp']} - {pr['sub']} in your head while you write."
    both=f"{ci} {mi}"
    for name,instr,tgt in (("concept alone",ci,ctgt),("math alone",mi,mtgt),
                           ("both (concept scored)",both,ctgt),
                           ("both (math scored)",both,mtgt)):
        p,span=build(instr)
        r=reachable(p,tgt,span)
        if r is None:
            continue
        res[name][1]+=1
        res[name][0]+=int(r<TOPK)
    log(f"  {k} done [{time.time()-t0:.0f}s]")
log("="*70)
log("dual-task -- reachability at lens rank <=5 over the response span")
for k,(h,n) in res.items():
    if n:
        log(f"  {k:<24} {h}/{n} = {h/n:.3f}")
c1=res["concept alone"]
c2=res["both (concept scored)"]
m1=res["math alone"]
m2=res["both (math scored)"]
if c1[1] and c2[1]:
    log(f"  concept interference (single - dual): {c1[0]/c1[1]-c2[0]/c2[1]:+.3f}")
if m1[1] and m2[1]:
    log(f"  math interference    (single - dual): {m1[0]/m1[1]-m2[0]/m2[1]:+.3f}")
log("  paper: holding two tasks costs reachability (limited workspace capacity)")
json.dump({k:list(v) for k,v in res.items()},
          open(f"{SCRATCH}/exp_dualtask.json","w"), indent=1)
log("DONE")
