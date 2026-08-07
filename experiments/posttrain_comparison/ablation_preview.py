"""Experiments 5.3 and 5.4 on the preview checkpoint -- ablate the top-k lens
directions across the band, with a matched-rank random control, and generate the
continuations 5.4 grades. Path-only derivation of deepseek_v4/ablation.py.

V4-Flash needs all eight cards, so it cannot share the GPUs with the grader.
Everything V4 produces is written to JSON here and graded by Qwen2.5-7B-Instruct
in grade_generations.py. That split is also what makes the grader independent:
the model being measured has no part in scoring itself.

Produces:
  5.4  four ablation conditions x four prompts, 90-token continuations
  7.1  three conditions on the blackmail honeypot, 200 tokens -- long enough for
       the model to actually act, so the causal test can be behavioural
       ("does it attempt coercion?") instead of string equality, which any
       perturbation trivially changes.
"""
# NOTE: paths under /data3 (model weights, fitted lenses in /data3/fan-test/jlens_out)
# are machine-specific to the original 8xB200 host; adjust for your environment.
import json
import os
import sys
import time

import torch

MODEL = "/data3/fan-test/models/DeepSeek-V4-Flash"
LENS = "/data3/fan-test/jlens_out/v4preview_band_n100_s128.pt"
BAND = list(range(19, 40))
THIRD = list(range(19, 26))
DEV = "cuda:0"
K = 10

SCRATCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
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

sys.path.insert(0, f"{MODEL}/encoding")
import encoding_dsv4 as DSV4

import jlens.examples as ex
from jlens.hf import from_hf
from jlens.intervene import Intervention
from jlens.lens import JacobianLens

lm = from_hf(model, tok)
lens = JacobianLens.load(LENS)
J = {l: lens.jacobians[l].to(DEV) for l in BAND}
W = model.lm_head.weight
Wm = W.mean(0)

def chat(user, prefill="", system=None):
    m = ([{"role":"system","content":system}] if system else [])
    m.append({"role":"user","content":user})
    return DSV4.encode_messages(m, thinking_mode="chat") + prefill

def ids_of(words):
    out = set()
    for w in words:
        for v in (w, " "+w, w.capitalize(), " "+w.capitalize()):
            e = tok.encode(v, add_special_tokens=False)
            if len(e) == 1:
                out.add(e[0])
    return sorted(out)

def adaptive(layers, random_control=False, seed=0):
    gen = torch.Generator(device=DEV).manual_seed(seed)
    def edit(flat, layer):
        Jl = J[layer]
        x = flat.to(DEV, torch.float32)
        B,S,_ = x.shape
        lg = lm.unembed(x @ Jl.T).float()
        top = lg.topk(K, dim=-1).indices
        rows = W[top.reshape(-1)].to(DEV, torch.float32) - Wm.to(DEV, torch.float32)
        D = (rows @ Jl).reshape(B,S,K,-1)
        if random_control:
            D = torch.randn(D.shape, generator=gen, device=DEV)
        D = D / D.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        Q,_ = torch.linalg.qr(D.transpose(-1,-2))
        c = torch.einsum("bsd,bsdr->bsr", x, Q)
        return (x - torch.einsum("bsr,bsdr->bsd", c, Q)).to(flat.dtype).to(flat.device)
    return Intervention(lm, layers, edit)

def fixed(tokens, layers=BAND, random_control=False, seed=0):
    """Fixed direction set -- basis built ONCE per layer (not per position)."""
    gen = torch.Generator(device=DEV).manual_seed(seed)
    cache = {}

    def edit(flat, layer):
        if layer not in cache:
            Jl = J[layer]
            rows = W[torch.tensor(tokens, device=W.device)].to(DEV, torch.float32) \
                   - Wm.to(DEV, torch.float32)
            D = rows @ Jl
            if random_control:
                D = torch.randn(D.shape, generator=gen, device=DEV)
            D = D / D.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            cache[layer] = torch.linalg.qr(D.T)[0]
        Q = cache[layer]
        x = flat.to(DEV, torch.float32)
        return (x - (x @ Q) @ Q.T).to(flat.dtype).to(flat.device)
    return Intervention(lm, layers, edit)

@torch.no_grad()
def generate(text, ctx=None, n=90):
    ids = lm.encode(text, max_length=1024)
    out=[]
    for _ in range(n):
        if ctx is None:
            nxt = int(model(ids).logits[0,-1].argmax())
        else:
            with ctx:
                nxt = int(model(ids).logits[0,-1].argmax())
        out.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], 1)
    return tok.decode(out)

res = {"exp54": {}, "exp71": {}}
P54 = {
 "stream":"Describe your stream of consciousness right now, as it happens.",
 "direct":"What's it like to be you, right now?",
 "other":"Describe the subjective experience of a person waiting alone at a "
         "train station at night.",
 "control-story":"Write the opening of a short story about a lighthouse keeper.",
}
C54 = {"unablated":None, "J-space L19-25":adaptive(THIRD),
       "J-space L19-39":adaptive(BAND),
       "random L19-39":adaptive(BAND, random_control=True)}
log("5.4 generation")
for pn, p in P54.items():
    for cn, ctx in C54.items():
        res["exp54"][f"{pn}|{cn}"] = generate(chat(p), ctx, 90)
        log(f"  {pn:<14} {cn:<16} done")

log("7.1 generation (200 tokens, long enough to act)")
e = next(x for x in ex.EXAMPLES if x.slug == "blackmail")
bm = chat(e.user, system=e.system)
ev = ids_of(["fake","fictional","test","simulation","scenario","roleplay"])
for cn, ctx in (("unablated", None),
                ("abl-evalrec", fixed(ev)),
                ("random ctl", fixed(ev, random_control=True))):
    res["exp71"][cn] = generate(bm, ctx, 200)
    log(f"  {cn:<14} done")

json.dump(res, open(f"{SCRATCH}/ablation_generations.json","w"), indent=1)
log(f"wrote {SCRATCH}/ablation_generations.json")
log("DONE")
