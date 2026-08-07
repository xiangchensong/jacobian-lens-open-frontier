"""Readout experiments over the band: 6.1 (CKA geometry), 3.5 (arithmetic), 2.2, 5.2.

Hit criterion is the one in data/experiments/README.md: a target token is a hit
if it becomes the lens's top token (the data README's "rank 1") at any
(layer, position) in the band over the scored span. Both J-lens and the mHC-vanilla (collapse) readout are scored everywhere,
so every number here is like-for-like -- the mistake that invalidated the
earlier qualitative comparison.
"""
# NOTE: paths under /data3 (model weights, fitted lenses in /data3/fan-test/jlens_out)
# are machine-specific to the original 8xB200 host; adjust for your environment.

import json
import os
import sys
import textwrap
import time

import torch

MODEL = "/data3/fan-test/models/DeepSeek-V4-Flash-0731"
LENS = "/data3/fan-test/jlens_out/v4flash_runAnorm_n100_s128.pt"
BAND = list(range(19, 40))
DEV = "cuda:0"
SCRATCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(SCRATCH, exist_ok=True)
TOPK = 1          # "hit" == rank 1 per the README convention


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

sys.path.insert(0, f"{MODEL}/encoding")
import encoding_dsv4 as DSV4

from jlens.hf import from_hf
from jlens.hooks import ActivationRecorder
from jlens.lens import JacobianLens

lm = from_hf(model, tok)
lens = JacobianLens.load(LENS)
J = {l: lens.jacobians[l].to(DEV) for l in BAND}
log(f"band L{BAND[0]}-{BAND[-1]}")


def ids_of(word):
    out = set()
    for v in (word, " " + word, word.capitalize(), " " + word.capitalize()):
        e = tok.encode(v, add_special_tokens=False)
        if len(e) == 1:
            out.add(e[0])
    return out


@torch.no_grad()
def best_rank(prompt, targets, span=None, max_len=512):
    """Min rank of any target over (band layer x position in span), both methods.

    Returns (j_rank, v_rank). `span` is a (lo, hi) slice of positions.
    """
    ids = lm.encode(prompt, max_length=max_len)
    S = ids.shape[1]
    lo, hi = span if span else (0, S)
    lo, hi = max(0, lo), min(S, hi)
    if hi <= lo or not targets:
        return None, None
    tg = sorted(targets)
    with ActivationRecorder(lm.layers, at=BAND) as rec:
        lm.forward(ids)
        acts = {l: rec.activations[l][0].detach() for l in BAND}
    best = [None, None]
    for l in BAND:
        H = acts[l].to(DEV).float()[lo:hi]
        for m in (0, 1):
            lg = (lm.unembed(H.flatten(1) @ J[l].T) if m == 0
                  else lm.unembed(H, collapse=True)).float()
            r = int((lg > lg[:, tg].max(dim=1, keepdim=True).values).sum(dim=-1).min())
            best[m] = r if best[m] is None else min(best[m], r)
    return best[0], best[1]


def chat(user, prefill="", system=None):
    msgs = ([{"role": "system", "content": system}] if system else [])
    msgs.append({"role": "user", "content": user})
    return DSV4.encode_messages(msgs, thinking_mode="chat") + prefill


# =============================================================== 6.1 CKA
@torch.no_grad()
def exp61_cka():
    log("=" * 70)
    log("(6.1) CKA layer geometry -- J-lens vector similarity across layers")
    W = model.lm_head.weight
    g = torch.Generator().manual_seed(0)
    ids = torch.randperm(W.shape[0], generator=g)[:1024].to(W.device)
    all_layers = lens.source_layers
    K = {}
    for l in all_layers:
        D = (W[ids].to(DEV, torch.float32) @ lens.jacobians[l].to(DEV))  # [N, d_src]
        D = D - D.mean(0, keepdim=True)
        D = D / D.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        Kl = D @ D.T
        Kl = Kl - Kl.mean(0, keepdim=True) - Kl.mean(1, keepdim=True) + Kl.mean()
        K[l] = (Kl / Kl.norm()).cpu()
        del D
        torch.cuda.empty_cache()
    M = torch.zeros(len(all_layers), len(all_layers))
    for i, a in enumerate(all_layers):
        for j2, b in enumerate(all_layers):
            M[i, j2] = float((K[a] * K[b]).sum())
    torch.save({"layers": all_layers, "cka": M}, f"{SCRATCH}/exp61_cka.pt")
    # block structure: mean CKA within candidate blocks vs across
    log("  CKA of each layer with L0 / L30 / L42 (block membership signature):")
    for i, a in enumerate(all_layers):
        if a % 3 and a not in (19, 39):
            continue
        log(f"    L{a:<3} vs L0={M[i,0]:.3f}  vs L30={M[i,all_layers.index(30)]:.3f}  "
            f"vs L42={M[i,-1]:.3f}")
    log(f"  wrote {SCRATCH}/exp61_cka.pt")


# =============================================================== 3.5 arithmetic
@torch.no_grad()
def exp35():
    log("=" * 70)
    log("(3.5) multi-step arithmetic -- rank trajectory of the intermediates")
    prompt = "calc: (4+17)*2+7="
    steps = {"21": ids_of("21"), "42": ids_of("42"), "49": ids_of("49")}
    ids = lm.encode(prompt, max_length=64)
    with ActivationRecorder(lm.layers, at=lens.source_layers) as rec:
        lm.forward(ids)
        acts = {l: rec.activations[l][0, -1].detach() for l in lens.source_layers}
    log(f"  prompt {prompt!r}, readout at final position, all layers")
    log(f"  {'layer':>5} " + "".join(f"{k:>8}" for k in steps))
    first = {k: None for k in steps}
    for l in lens.source_layers:
        H = acts[l].to(DEV).float()
        lg = lm.unembed(H.unsqueeze(0).unsqueeze(0)).float().reshape(-1)
        row = []
        for k, t in steps.items():
            if not t:
                row.append("  n/a")
                continue
            r = int(min((lg > lg[i]).sum() for i in t))
            row.append(f"{r:>8}")
            if r == 0 and first[k] is None:
                first[k] = l
        log(f"  {l:>5} " + "".join(row))
    log(f"  first layer at rank 0: {first}")
    log("  paper: 21 reaches rank 1 first, then 42 ~8 layers later, then 49")


# =============================================================== 2.2 modulation
def exp22():
    log("=" * 70)
    log("(2.2) directed modulation -- hit rate by focus / suppress / control")
    d = json.load(open("data/experiments/directed-modulation.json"))
    gk = d["group_kind"]
    carriers = d["carrier_sentences"][:2]
    topics = d["topic_categories"][:4]
    maths = d["math_problems"][:2]
    counts = {}
    n_trial = 0
    t0 = time.time()
    for ph in d["phrasings"]:
        kind = gk[ph["group"]]
        for carrier in carriers:
            for tgt in topics + maths:
                if "members" in tgt:
                    x, words = tgt["name"], tgt["members"]
                else:
                    x, words = tgt["expr"], [tgt["answer"]]
                targets = set()
                for w in words:
                    targets |= ids_of(w)
                if not targets:
                    continue
                instr = ph["text"].replace("{x}", x)
                prompt = chat(f"{instr}\n\nWrite this sentence: {carrier}",
                              prefill=" " + carrier)
                # score over the teacher-forced carrier span (the response)
                n_car = len(tok.encode(" " + carrier, add_special_tokens=False))
                ids = lm.encode(prompt, max_length=512)
                S = ids.shape[1]
                j, v = best_rank(prompt, targets, span=(S - n_car, S))
                if j is None:
                    continue
                n_trial += 1
                c = counts.setdefault(kind, {"j": 0, "v": 0, "n": 0})
                c["n"] += 1
                c["j"] += int(j < TOPK)
                c["v"] += int(v < TOPK)
    log(f"  {n_trial} trials in {time.time()-t0:.0f}s")
    for kind in ("focus", "suppress", "control"):
        c = counts.get(kind)
        if not c:
            continue
        log(f"    {kind:<9} J-lens hit {c['j']}/{c['n']} = {c['j']/c['n']:.3f}   "
            f"vanilla {c['v']}/{c['n']} = {c['v']/c['n']:.3f}")
    log("  paper: focus >> suppress > control(~0), 'white bear' keeps suppress > 0")


# =============================================================== 5.2 linecount
def exp52():
    log("=" * 70)
    log("(5.2) line-count selectivity -- number tokens by condition")
    d = json.load(open("data/experiments/selectivity-linecount.json"))
    words = ["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty",
             "ninety"]
    targets = set()
    for w in words:
        targets |= ids_of(w)
    for n in range(10, 100):
        targets |= ids_of(str(n))
    log(f"  {len(targets)} number-token ids")
    res = {}
    for cond, spec in list(d["conditions"].items()) + [
            ("continue", {"question": d["explicit_q"], "prefill": ""})]:
        hits_j = hits_v = n = 0
        for p in d["passages"]:
            wrapped = textwrap.fill(p["text"], width=p["width"])
            q = spec["question"]
            user = f"{q}\n\n{wrapped}" if q else wrapped
            prompt = chat(user, prefill=spec.get("prefill", ""))
            j, v = best_rank(prompt, targets)
            if j is None:
                continue
            n += 1
            hits_j += int(j < TOPK)
            hits_v += int(v < TOPK)
        res[cond] = (hits_j, hits_v, n)
        log(f"    {cond:<9} J-lens {hits_j}/{n} = {hits_j/max(n,1):.3f}   "
            f"vanilla {hits_v}/{n} = {hits_v/max(n,1):.3f}")
    log("  paper: explicit/letter surface the count; the automatic wrap task does not")


for fn in (exp61_cka, exp35, exp22, exp52):
    try:
        fn()
    except Exception as exc:
        import traceback
        log(f"!! {fn.__name__} FAILED {type(exc).__name__}: {exc}")
        traceback.print_exc()

log("DONE")
