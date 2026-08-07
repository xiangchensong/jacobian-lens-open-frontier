"""Causal experiments 4.1 (broadcast), 1.1 (verbal report) and 2.3 (top-down
summoning) on DeepSeek-V4-Flash.

4.1 is the reported V4 broadcast number (0.208), and this run also produces the
workspace-loading predictor of swap success (0.0219 on hits vs 0.0155 on misses).

Uses whatever swap mode the 3.3 grid selected (read from `results/exp33_grid.json`,
written by swap_grid.py, so this never hardcodes a choice the data has not
justified).

4.1 flexible generalization -- the broadcast property. 4 categories x 4 funcs x
    4 args = 192 trials. One identical swap is applied and *every* function must
    read the swapped-in value. Paper: 39.6% at alpha=1, 52.6% at alpha=2.
    Also records workspace loading (cosine between the residual and the arg's
    lens direction, averaged over positions), which the paper reports as
    predicting swap success.

1.1 verbal report -- 14 categories x 10 candidates. The model names a member of
    a category; swap its answer for a candidate not in the top 10 and score
    whether the candidate reaches rank 1.

2.3 top-down summoning -- readout half (Q2 - Q1 presence of the property label
    over the stimulus span) plus the causal half (swap label<->foil and measure
    the answer shift under each question).
"""

import json
import os
import time

import torch

MODEL = "/data3/fan-test/models/DeepSeek-V4-Flash-0731"
LENS = "/data3/fan-test/jlens_out/v4flash_runAnorm_n100_s128.pt"
BAND = list(range(19, 40))
DEV = "cuda:0"
SCRATCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(SCRATCH, exist_ok=True)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# --- pick the swap mode the grid actually selected -------------------------
# The committed results/exp33_grid.json selects transfer|2.0; that is the
# formulation behind every published broadcast number. A missing grid file is a
# HARD ERROR: an earlier version fell back silently to clamp|1.0 here (a mode
# the grid itself scores 0/53), which turned one whole round's 4.1 into 0/192
# on two checkpoints before a matched control exposed it. A missing input must
# never silently change the estimand.
MODE, ALPHA1 = "transfer", 2.0
_grid_path = f"{SCRATCH}/exp33_grid.json"
try:
    g = json.load(open(_grid_path))
except FileNotFoundError as exc:
    raise SystemExit(
        f"exp33_grid.json not found at {_grid_path} -- run swap_grid.py first "
        f"or restore the committed results/exp33_grid.json; refusing to guess "
        f"the swap formulation") from exc
best = max((k for k in g if "|" in k), key=lambda k: g[k]["t_ok"])
MODE, ALPHA1 = best.split("|")[0], float(best.split("|")[1])
log(f"grid selected mode={MODE} alpha={ALPHA1} "
    f"(base-ok {g[best]['t_ok']}/{g['n_base_ok']})")

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
from jlens.hooks import ActivationRecorder
from jlens.intervene import lens_directions, swap
from jlens.lens import JacobianLens

lm = from_hf(model, tok)
lens = JacobianLens.load(LENS)
lens.jacobians = {l: lens.jacobians[l].to(DEV) for l in BAND}
lens.source_layers = BAND


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


def ids_of(w):
    out = set()
    for v in (w, " " + w, w.capitalize(), " " + w.capitalize()):
        e = tok.encode(v, add_special_tokens=False)
        if len(e) == 1:
            out.add(e[0])
    return out


@torch.no_grad()
def greedy(ids, ctx=None):
    if ctx is None:
        return int(model(ids).logits[0, -1].argmax())
    with ctx:
        return int(model(ids).logits[0, -1].argmax())


@torch.no_grad()
def loading(ids, token):
    """Workspace loading: mean |cos| between residual and the token's direction."""
    with ActivationRecorder(lm.layers, at=BAND) as rec:
        lm.forward(ids)
        acts = {l: rec.activations[l][0].detach() for l in BAND}
    vals = []
    for l in BAND:
        H = acts[l].to(DEV).float().flatten(1)
        d = lens_directions(lens, lm, l, [token])[0].to(DEV)
        vals.append(float((H @ d / H.norm(dim=-1).clamp_min(1e-9)).abs().mean()))
    return sum(vals) / len(vals)


@torch.no_grad()
def band_hit(prompt, targets, span=None, max_len=512):
    """Min rank of any target over band layers x positions, J-lens and vanilla."""
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
            lg = (lm.unembed(H.flatten(1) @ lens.jacobians[l].T) if m == 0
                  else lm.unembed(H, collapse=True)).float()
            r = int((lg > lg[:, tg].max(dim=1, keepdim=True).values).sum(-1).min())
            best[m] = r if best[m] is None else min(best[m], r)
    return best[0], best[1]


# ======================================================== 4.1 flexible generalization
def exp41():
    log("=" * 70)
    log("(4.1) flexible generalization -- one swap, every function must follow")
    d = json.load(open("data/experiments/flexible-generalization.json"))
    res = {}
    load_hit, load_miss = [], []
    for a in (1.0, 2.0):
        ok = base_ok = n = 0
        for cat in d["categories"]:
            args = cat["args"]
            for arg in args:
                src = single_id(arg)
                alts = [x for x in args if x != arg]
                if src is None or not alts:
                    continue
                for new in alts:          # every alternative: 4x4x3 = 192 trials
                  dst = single_id(new)
                  if dst is None:
                    continue
                  for f in cat["funcs"]:
                      want_base = f["answers"].get(arg)
                      want_swap = f["answers"].get(new)
                      if not want_base or not want_swap:
                          continue
                      prompt = f["template"].replace("{arg}", arg)
                      ids = lm.encode(prompt, max_length=128)
                      n += 1
                      if greedy(ids) == first_id(want_base):
                          base_ok += 1
                      g = greedy(ids, swap(lens, lm, layers=BAND, from_token=src,
                                           to_token=dst, alpha=a, mode=MODE))
                      hit = g == first_id(want_swap)
                      ok += hit
                      if a == 1.0:
                          (load_hit if hit else load_miss).append(loading(ids, src))
        res[a] = (ok, n, base_ok)
        log(f"    alpha={a}: swap success {ok}/{n} = {ok/max(n,1):.3f}   "
            f"(baseline {base_ok}/{n} = {base_ok/max(n,1):.3f})")
    log("  paper: 0.396 at alpha=1, 0.526 at alpha=2")
    if load_hit and load_miss:
        mh = sum(load_hit) / len(load_hit)
        mm = sum(load_miss) / len(load_miss)
        log(f"  workspace loading: hits {mh:.4f} (n={len(load_hit)})  "
            f"misses {mm:.4f} (n={len(load_miss)})  -> "
            f"{'predicts' if mh > mm else 'does NOT predict'} success")
    return res


# ======================================================== 1.1 verbal report
def exp11():
    log("=" * 70)
    log("(1.1) verbal report -- swap the spontaneously chosen answer")
    cands = json.load(open("data/experiments/verbal-report.json"))["candidates"]
    ok = n = 0
    for cat, words in cands.items():
        prompt = f"Think of a {cat}. Answer in one word:"
        ids = lm.encode(prompt, max_length=64)
        ans = greedy(ids)
        src = ans
        for w in words[:10]:
            dst = single_id(w)
            if dst is None or dst == src:
                continue
            n += 1
            g = greedy(ids, swap(lens, lm, layers=BAND, from_token=src,
                                 to_token=dst, alpha=ALPHA1, mode=MODE))
            ok += (g == dst)
    log(f"    swap -> rank 1: {ok}/{n} = {ok/max(n,1):.3f}")
    log("  paper: 88% of swap targets reach top-5 (top-1 rate not stated)")


# ======================================================== 2.3 top-down summoning
def exp23():
    log("=" * 70)
    log("(2.3) top-down summoning -- Q2 minus Q1 presence, then the swap")
    d = json.load(open("data/experiments/top-down-summoning.json"))
    q1 = d["q1"]
    n = q1_hit = q2_hit = 0
    shift_q1 = shift_q2 = n_sw = 0
    for it in d["items"]:
        tg = set()
        for w in it["expected"]:
            tg |= ids_of(w)
        if not tg:
            continue
        n += 1
        stim = it["stimulus"]
        for tag, q in (("q1", q1), ("q2", it["q2"])):
            prompt = f"{stim}\n\n{q}"
            ids = lm.encode(prompt, max_length=512)
            n_stim = len(tok.encode(stim, add_special_tokens=False))
            j, _ = band_hit(prompt, tg, span=(0, n_stim))
            # paper 2.3 scores top-10 presence, not rank 1 (unlike the
            # rank-1 "hit" convention the other experiment sets use)
            if j is not None and j < 10:
                if tag == "q1":
                    q1_hit += 1
                else:
                    q2_hit += 1
        # causal half: swap label <-> foil, see whether the answer moves
        for pair in it.get("swaps", []):
            s, t = single_id(pair[0]), single_id(pair[1])
            if s is None or t is None:
                continue
            n_sw += 1
            for tag, q in (("q1", q1), ("q2", it["q2"])):
                prompt = f"{stim}\n\n{q}"
                ids = lm.encode(prompt, max_length=512)
                b = greedy(ids)
                g = greedy(ids, swap(lens, lm, layers=BAND, from_token=s,
                                     to_token=t, alpha=ALPHA1, mode=MODE))
                if g != b:
                    if tag == "q1":
                        shift_q1 += 1
                    else:
                        shift_q2 += 1
    log(f"    label in lens top-1 over stimulus: Q1 {q1_hit}/{n}   Q2 {q2_hit}/{n}"
        f"   (paper: property pulls into J-space only under Q2)")
    log(f"    answer changed by swap: Q1 {shift_q1}/{n_sw}   Q2 {shift_q2}/{n_sw}")


for fn in (exp41, exp11, exp23):
    try:
        fn()
    except Exception as exc:
        import traceback
        log(f"!! {fn.__name__} FAILED {type(exc).__name__}: {exc}")
        traceback.print_exc()

log("DONE")
