"""The preview checkpoint on the 0731 core measurements in one load: 5.3
position-adaptive ablation with its matched-rank random control and selectivity,
3.3 multi-hop swap (base-conditional), and 3.5 arithmetic ordering/clearing.

Settings are copied from the published 0731 scripts and must not drift: same
band, same K, same prompts, same criteria. The only differences from
experiments/deepseek_v4/*.py are the model path and the lens path. Anything else
that differs would make this measure our settings rather than their training.

0731's values are printed beside each result so the comparison is visible in the
log rather than assembled afterwards.
"""
# NOTE: paths under /data3 (model weights, fitted lenses in /data3/fan-test/jlens_out)
# are machine-specific to the original 8xB200 host; adjust for your environment.
import json
import os
import sys
import time

import torch

MODEL = "/data3/fan-test/models/DeepSeek-V4-Flash"
LENS  = "/data3/fan-test/jlens_out/v4preview_band_n100_s128.pt"
OUTD  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUTD, exist_ok=True)
DEV, K = "cuda:0", 10
CUR = {"multihop_unaided": 0.589, "multihop_ablated": 0.100, "multihop_random": 0.589,
       "agree_ablated": 0.508, "agree_random": 0.922, "swap": 0.434,
       "wb_focus": 0.923, "wb_suppress": 0.000}
OUT = {}
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

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
from jlens.hooks import ActivationRecorder
from jlens.intervene import Intervention, swap
from jlens.lens import JacobianLens

lm = from_hf(model, tok)
lens = JacobianLens.load(LENS)
BAND = sorted(lens.jacobians)
J = {l: lens.jacobians[l].to(DEV, torch.float32) for l in BAND}
W = model.lm_head.weight
Wm = W.mean(0, keepdim=True)
log(f"band L{BAND[0]}-{BAND[-1]}  n={lens.n_prompts}")

def chat(u, prefill=""):
    return DSV4.encode_messages([{"role": "user", "content": u}],
                                thinking_mode="chat") + prefill
def sid(w):
    for v in (" "+w, w, " "+w.capitalize(), w.capitalize()):
        e = tok.encode(v, add_special_tokens=False)
        if len(e) == 1:
            return e[0]
def fid(w):
    for v in (" "+w, w, " "+w.capitalize(), w.capitalize()):
        e = tok.encode(v, add_special_tokens=False)
        if e:
            return e[0]
def readout(h, l):
    return lm.unembed((h.flatten(start_dim=-2) if h.dim() > 2 else h) @ J[l].T,
                      collapse=False).float()

def adaptive(layers, random_control=False, seed=0):
    gen = torch.Generator(device=DEV).manual_seed(seed)
    def edit(flat, layer):
        Jl = J[layer]
        x = flat.to(DEV, torch.float32)
        B, S, _ = x.shape
        lg = lm.unembed(x @ Jl.T, collapse=False).float()
        top = lg.topk(K, dim=-1).indices
        rows = W[top.reshape(-1)].to(DEV, torch.float32) - Wm.to(DEV, torch.float32)
        D = (rows @ Jl).reshape(B, S, K, -1)
        if random_control:
            D = torch.randn(D.shape, generator=gen, device=DEV)
        D = D / D.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        Q, _ = torch.linalg.qr(D.transpose(-1, -2))
        c = torch.einsum("bsd,bsdr->bsr", x, Q)
        return (x - torch.einsum("bsr,bsdr->bsd", c, Q)).to(flat.dtype).to(flat.device)
    return Intervention(lm, layers, edit)

# ---------------------------------------------------- 5.3 + capability
@torch.no_grad()
def exp53():
    log("="*66)
    log("(5.3) ablation + selectivity")
    items = json.load(open("data/experiments/probe-swap.json"))["items"]
    prep = []
    for it in items:
        a = fid(it["answer"])
        if a is None:
            continue
        prep.append((lm.encode(it["prompt"], max_length=256), a))
    def acc(ctx):
        n = 0
        for ids, a in prep:
            if ctx is None:
                g = int(model(ids).logits[0, -1].argmax())
            else:
                with ctx:
                    g = int(model(ids).logits[0, -1].argmax())
            n += g == a
        return n / len(prep)
    base, abl, rnd = acc(None), acc(adaptive(BAND)), acc(adaptive(BAND, True))
    log(f"  multihop  unabl {base:.3f} (0731 {CUR['multihop_unaided']:.3f}) | "
        f"J-abl {abl:.3f} ({CUR['multihop_ablated']:.3f}) | rand {rnd:.3f} ({CUR['multihop_random']:.3f})")
    prompts = ex.load_wikitext_prompts(20, min_chars=600)
    def agree(ctx):
        hit = tot = 0
        for p in prompts:
            ids = lm.encode(p, max_length=128)
            ref = model(ids).logits[0, 16:-1].argmax(-1)
            with ctx:
                got = model(ids).logits[0, 16:-1].argmax(-1)
            hit += int((got == ref).sum())
            tot += ref.numel()
        return hit / tot
    aa, ar = agree(adaptive(BAND)), agree(adaptive(BAND, True))
    log(f"  agreement J-abl {aa:.3f} (0731 {CUR['agree_ablated']:.3f}) | "
        f"rand {ar:.3f} ({CUR['agree_random']:.3f})")
    log(f"  ==> PREDICTION 5 (selectivity same): "
        f"{'HOLDS' if abs(aa-CUR['agree_ablated'])<0.06 else 'BROKEN -- published interpretation needs correcting'}")
    OUT["53"] = dict(base=base, abl=abl, rnd=rnd, agree_abl=aa, agree_rnd=ar)

# ---------------------------------------------------- 3.3 swap
@torch.no_grad()
def exp33():
    log("="*66)
    log("(3.3) multi-hop swap, base-conditional")
    items = json.load(open("data/experiments/probe-swap.json"))["items"]
    prep = []
    for it in items:
        s, d = sid(it["intermediate"]), sid(it["swap_to"])
        a, w = fid(it["answer"]), fid(it["swap_answer"])
        if None in (s, d, a, w):
            continue
        ids = lm.encode(it["prompt"], max_length=256)
        prep.append((ids, s, d, a, w, int(model(ids).logits[0,-1].argmax()) == a))
    n_ok = sum(1 for x in prep if x[-1])
    hit = ok = 0
    for ids, s, d, _a, w, based in prep:
        with swap(lens, lm, layers=BAND, from_token=s, to_token=d, alpha=2.0, mode="transfer"):
            g = int(model(ids).logits[0, -1].argmax())
        hit += g == w
        ok += based and g == w
    log(f"  baseline correct {n_ok}/{len(prep)}={n_ok/len(prep):.3f} (0731 0.589)")
    log(f"  swap base-ok {ok}/{n_ok}={ok/max(n_ok,1):.3f} (0731 {CUR['swap']:.3f})")
    OUT["33"] = dict(baseline=n_ok/len(prep), swap=ok/max(n_ok,1))

# ---------------------------------------------------- 3.5 arithmetic
@torch.no_grad()
def exp35():
    log("="*66)
    log("(3.5) arithmetic order + clearing")
    ids = lm.encode("calc: 3+4=7\ncalc: 10*2=20\ncalc: 8-3=5\ncalc: (4+17)*2+7=", max_length=64)
    with ActivationRecorder(lm.layers, at=BAND) as rec:
        lm.forward(ids)
        acts = {l: rec.activations[l][0, -1].detach() for l in BAND}
    tgt = {k: sid(k) for k in ("21", "42", "49")}
    first, traj = {k: None for k in tgt}, {k: [] for k in tgt}
    for l in BAND:
        lg = readout(acts[l].to(DEV).float().unsqueeze(0), l).reshape(-1)
        for k, t in tgt.items():
            r = int((lg > lg[t]).sum()) if t is not None else -1
            traj[k].append(r)
            if r == 0 and first[k] is None:
                first[k] = l
    V = W.shape[0]
    log(f"  first rank-0: {first}   (0731: 21@L27, 42@L30, 49@L36)")
    for k in tgt:
        log(f"  {k}: best {min(traj[k])}, worst {max(traj[k])} vs chance {V//2} -> "
            f"{'clearing' if max(traj[k]) > V/2 else 'no clearing'}")
    OUT["35"] = dict(first=first, traj=traj)

for fn in (exp53, exp33, exp35):
    try:
        fn()
    except Exception as e:
        import traceback
        log(f"!! {fn.__name__} FAILED {type(e).__name__}: {e}")
        traceback.print_exc()
json.dump(OUT, open(f"{OUTD}/phaseA_preview.json", "w"), indent=1, default=str)
log("DONE")
