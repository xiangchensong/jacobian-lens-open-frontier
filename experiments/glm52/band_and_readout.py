"""GLM-5.2: band validation (6.2, 6.1), arithmetic (3.5) and the multi-hop swap (3.3).

Band caveat, stated up front: the lens covers only the PROJECTED band L34-71, so
6.2 here can check whether the four statistics behave band-like *within* the
window and whether turning points appear at its edges -- it cannot see an onset
earlier than L34. If the stats are already elevated at the L34 edge (no visible
onset), the projection was too narrow and a wider pilot fit is required before
trusting downstream numbers.
"""
# NOTE: paths under /data3 (model weights, fitted lenses in /data3/fan-test/jlens_out)
# are machine-specific to the original 8xB200 host; adjust for your environment.
import json
import os
import time

import torch

S = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(S, exist_ok=True)

import jlens.fp8_autograd  # noqa: F401  must precede model load: registers fp8 backward

MODEL = "/data3/fan-test/models/GLM-5.2-FP8"
LENS = "/data3/fan-test/jlens_out/glm52_band_n100_s128.pt"
DEV = "cuda:0"
SKIP = 16

def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

from transformers import AutoModelForCausalLM, AutoTokenizer

n_gpu = torch.cuda.device_count()
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype="auto", device_map="auto",
    max_memory={i: "146GiB" for i in range(n_gpu)},
    attn_implementation="eager", experts_implementation="grouped_mm")
model.eval()
tok = AutoTokenizer.from_pretrained(MODEL)
log("LOADED")

import jlens.examples as ex
from jlens.hf import from_hf
from jlens.hooks import ActivationRecorder
from jlens.intervene import swap
from jlens.lens import JacobianLens

lm = from_hf(model, tok)
lens = JacobianLens.load(LENS)
BAND = lens.source_layers
J = {l: lens.jacobians[l].to(DEV) for l in BAND}
log(f"lens: {lens!r}")


def single_id(w):
    for v in (" " + w, w, " " + w.capitalize(), w.capitalize()):
        e = tok.encode(v, add_special_tokens=False)
        if len(e) == 1:
            return e[0]


def first_id(w):
    for v in (" " + w, w, " " + w.capitalize(), w.capitalize()):
        e = tok.encode(v, add_special_tokens=False)
        if e:
            return e[0]


# ---------------------------------------------------- 6.2 within-window
@torch.no_grad()
def exp62(n_corpus=30):
    log("=" * 70)
    log(f"(6.2) band statistics WITHIN L{BAND[0]}-{BAND[-1]} (window caveat applies)")
    prompts = ex.load_wikitext_prompts(n_corpus, min_chars=600)
    acc = {l: {"t1": 0, "n": 0, "kurt": [], "auto": 0, "an": 0, "shuf": 0}
           for l in BAND}
    for pi, p in enumerate(prompts):
        ids = lm.encode(p, max_length=128)
        Sq = ids.shape[1]
        if Sq < SKIP + 4:
            continue
        nxt = ids[0, 1:].to(DEV)
        with ActivationRecorder(lm.layers, at=BAND) as rec:
            lm.forward(ids)
            acts = {l: rec.activations[l][0].detach() for l in BAND}
        for l in BAND:
            H = acts[l].to(DEV).float()
            lg = lm.unembed(H @ J[l].T).float()[SKIP:Sq - 1].to(DEV)
            gold = nxt[SKIP:Sq - 1]
            a = acc[l]
            a["t1"] += int((lg.argmax(-1) == gold).sum())
            a["n"] += lg.shape[0]
            mu = lg.mean(-1, keepdim=True)
            sd = lg.std(-1, keepdim=True).clamp_min(1e-6)
            a["kurt"].append(float((((lg - mu) / sd).pow(4).mean(-1) - 3).mean()))
            t1 = lg.argmax(-1)
            if t1.numel() > 1:
                a["auto"] += int((t1[1:] == t1[:-1]).sum())
                a["an"] += t1.numel() - 1
                perm = t1[torch.randperm(t1.numel(), device=t1.device)]
                a["shuf"] += int((perm[1:] == perm[:-1]).sum())
        if (pi + 1) % 10 == 0:
            log(f"  corpus {pi+1}/{n_corpus}")
    log(f"  {'layer':>5} {'top1':>8} {'kurt':>8} {'ex-auto':>8}")
    rows = []
    for l in BAND:
        a = acc[l]
        if not a["n"]:
            continue
        t1 = a["t1"] / a["n"]
        ku = sum(a["kurt"]) / len(a["kurt"])
        ea = a["auto"] / max(a["an"], 1) - a["shuf"] / max(a["an"], 1)
        rows.append(dict(layer=l, top1=t1, kurt=ku, exauto=ea))
        log(f"  {l:>5} {t1:>8.4f} {ku:>8.2f} {ea:>8.4f}")
    json.dump(rows, open(f"{S}/glm_exp62.json", "w"), indent=1)
    lo, hi = rows[0], rows[-1]
    log(f"  edge check: top1 L{BAND[0]}={lo['top1']:.4f} (DeepSeek onset ~0.006), "
        f"L{BAND[-1]}={hi['top1']:.4f} (DeepSeek offset jump ~0.17)")
    log("  if L34 already looks mid-band, the projected window missed the onset")


# ---------------------------------------------------- 6.1 CKA within-window
@torch.no_grad()
def exp61():
    log("=" * 70)
    log("(6.1) CKA between fitted layers")
    W = model.lm_head.weight
    g = torch.Generator().manual_seed(0)
    ids = torch.randperm(W.shape[0], generator=g)[:512].to(W.device)
    K = {}
    for l in BAND:
        D = (W[ids].to(DEV, torch.float32) @ J[l])
        D = D - D.mean(0, keepdim=True)
        D = D / D.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        Kl = D @ D.T
        Kl = Kl - Kl.mean(0, keepdim=True) - Kl.mean(1, keepdim=True) + Kl.mean()
        K[l] = (Kl / Kl.norm()).cpu()
        del D
        torch.cuda.empty_cache()
    a, b, c = BAND[0], BAND[len(BAND) // 2], BAND[-1]
    for l in BAND[::4]:
        log(f"    L{l:<3} vs L{a}={float((K[l]*K[a]).sum()):.3f}  "
            f"vs L{b}={float((K[l]*K[b]).sum()):.3f}  vs L{c}={float((K[l]*K[c]).sum()):.3f}")


# ---------------------------------------------------- 3.5 arithmetic
@torch.no_grad()
def exp35():
    log("=" * 70)
    log("(3.5) arithmetic: calc: (4+17)*2+7=   intermediates 21 -> 42 -> 49")
    ids = lm.encode("calc: (4+17)*2+7=", max_length=64)
    with ActivationRecorder(lm.layers, at=BAND) as rec:
        lm.forward(ids)
        acts = {l: rec.activations[l][0, -1].detach() for l in BAND}
    tgt = {k: [i for i in [single_id(k)] if i is not None] for k in ("21", "42", "49")}
    first = {k: None for k in tgt}
    log(f"  {'layer':>5} {'r21':>8} {'r42':>8} {'r49':>8}")
    for l in BAND:
        H = acts[l].to(DEV).float()
        lg = lm.unembed(H.unsqueeze(0) @ J[l].T).float().reshape(-1)
        row = []
        for k, t in tgt.items():
            if not t:
                row.append("n/a")
                continue
            r = int(min((lg > lg[i]).sum() for i in t))
            row.append(f"{r:>8}")
            if r == 0 and first[k] is None:
                first[k] = l
        log(f"  {l:>5} " + "".join(row))
    log(f"  first rank-0: {first}  (DeepSeek: 21@L27, 42@L30, 49@L36, in order)")


# ---------------------------------------------------- 3.3 multi-hop swap
def exp33():
    log("=" * 70)
    log("(3.3) multi-hop swap, 90 prompts, transfer mode alpha=2")
    items = json.load(open("data/experiments/probe-swap.json"))["items"]
    prep = []
    with torch.no_grad():
        for it in items:
            s, d = single_id(it["intermediate"]), single_id(it["swap_to"])
            a, w = first_id(it["answer"]), first_id(it["swap_answer"])
            if None in (s, d, a, w):
                continue
            ids = lm.encode(it["prompt"], max_length=256)
            base = int(model(ids).logits[0, -1].argmax())
            prep.append((ids, s, d, a, w, base == a))
    n_ok = sum(1 for x in prep if x[-1])
    log(f"  {len(prep)} scored, baseline correct {n_ok} "
        f"({n_ok/max(len(prep),1):.3f}; DeepSeek 0.589, Claude near-ceiling)")
    hit = ok = 0
    with torch.no_grad():
        for ids, s, d, _, w, base in prep:
            ctx = swap(lens, lm, layers=BAND, from_token=s, to_token=d,
                       alpha=2.0, mode="transfer")
            with ctx:
                g = int(model(ids).logits[0, -1].argmax())
            hit += g == w
            ok += base and g == w
    log(f"  swap success: all {hit}/{len(prep)}={hit/len(prep):.3f}   "
        f"base-ok {ok}/{n_ok}={ok/max(n_ok,1):.3f}")
    log("  compare: DeepSeek 0.434 base-ok; paper 0.54-0.70")
    log("  KEY QUESTION: single-stream GLM tests the stream-dilution hypothesis")


for fn in (exp62, exp61, exp35, exp33):
    try:
        fn()
    except Exception as exc:
        import traceback
        log(f"!! {fn.__name__} FAILED {type(exc).__name__}: {exc}")
        traceback.print_exc()
log("DONE")
