"""Re-run every lens-dependent experiment against the n=100, n=1000 and
disjoint-100 lenses side by side, and report each metric's n=1000 movement
against the disjoint-100 noise floor.

This is the test of the pre-registered prediction: the headline swap number comes
out 0.4340 -> 0.4340, and no metric moves further than corpus-sampling noise. The
disjoint-100 lens (prompts 900-1000) is built here if it is not already on disk,
because the n=1000 corpus *contains* the n=100 one, so a plain difference has no
scale without it.

Prediction on record before this ran: **no meaningful change**. The n-scaling
curve measured on 3.3 was flat from n=60 to n=100 (0.434 -> 0.434), so the
estimator had already converged and ten times the corpus should land in the same
place. If that holds, the "your lens is undertrained" objection to the swap
shortfalls is closed. If it does not hold, the n-scaling measurement was
misleading and that is the more interesting outcome -- either way it gets
reported as measured.

Runs: 3.3 multi-hop swap, 4.1 flexible generalization, 5.1 causal, 1.3
decomposition, 5.3 ablation, and the 6.2 band statistics (to check the band
itself does not move).
"""

import json
import os
import time
from collections import defaultdict

import torch

MODEL = "/data3/fan-test/models/DeepSeek-V4-Flash-0731"
LENSES = [
    ("n=100", "/data3/fan-test/jlens_out/v4flash_runAnorm_n100_s128.pt"),
    ("n=1000", "/data3/fan-test/jlens_out/v4flash_band_n1000_s128.pt"),
    # prompts 900-1000 only: a DISJOINT 100-prompt fit. Its distance from n=100
    # is corpus-sampling noise, and any n=1000 effect must exceed it to count.
    ("disjoint100", "/data3/fan-test/jlens_out/v4flash_disjoint100_s128.pt"),
]
BAND = list(range(19, 40))
DEV = "cuda:0"
K = 10
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

import sys

sys.path.insert(0, f"{MODEL}/encoding")
import encoding_dsv4 as DSV4

import jlens.examples as ex
from jlens.hf import from_hf
from jlens.hooks import ActivationRecorder
from jlens.intervene import Intervention, swap
from jlens.lens import JacobianLens

lm = from_hf(model, tok)
W = model.lm_head.weight
Wm = W.mean(0)
SKIP = 16


def chat(user, prefill=""):
    return DSV4.encode_messages([{"role": "user", "content": user}],
                                thinking_mode="chat") + prefill


def first_id(w):
    for v in (" " + w, w, " " + w.capitalize(), w.capitalize()):
        e = tok.encode(v, add_special_tokens=False)
        if e:
            return e[0]


def single_id(w):
    for v in (" " + w, w, " " + w.capitalize(), w.capitalize()):
        e = tok.encode(v, add_special_tokens=False)
        if len(e) == 1:
            return e[0]


@torch.no_grad()
def greedy(ids, ctx=None):
    if ctx is None:
        return int(model(ids).logits[0, -1].argmax())
    with ctx:
        return int(model(ids).logits[0, -1].argmax())


PROBE = json.load(open("data/experiments/probe-swap.json"))["items"]
FLEX = json.load(open("data/experiments/flexible-generalization.json"))
LANGS = json.load(open("data/experiments/selectivity-language.json"))
CORPUS = ex.load_wikitext_prompts(15, min_chars=600)


def run_for(tag, path):
    lens = JacobianLens.load(path)
    have = [l for l in BAND if l in lens.jacobians]
    lens.jacobians = {l: lens.jacobians[l].to(DEV) for l in have}
    lens.source_layers = have
    J = lens.jacobians
    out = {}
    log(f"===== {tag}  ({path.split('/')[-1]}, L{have[0]}-{have[-1]}, "
        f"n_prompts={lens.n_prompts})")

    # ---- 3.3
    prep = []
    for it in PROBE:
        s, d = single_id(it["intermediate"]), single_id(it["swap_to"])
        a, w = first_id(it["answer"]), first_id(it["swap_answer"])
        if None in (s, d, a, w):
            continue
        ids = lm.encode(it["prompt"], max_length=256)
        prep.append((ids, s, d, a, w, greedy(ids) == a))
    n_ok = sum(1 for p in prep if p[-1])
    hit = ok = 0
    for ids, s, d, _a, w, base in prep:
        g = greedy(ids, swap(lens, lm, layers=have, from_token=s, to_token=d,
                             alpha=2.0, mode="transfer"))
        hit += g == w
        ok += base and g == w
    out["3.3_all"] = hit / len(prep)
    out["3.3_base_ok"] = ok / max(n_ok, 1)
    log(f"  3.3 multi-hop swap   all {hit}/{len(prep)}={hit/len(prep):.3f}   "
        f"base-ok {ok}/{n_ok}={ok/max(n_ok,1):.3f}")

    # ---- 4.1
    for a_ in (1.0, 2.0):
        good = n = 0
        for cat in FLEX["categories"]:
            args = cat["args"]
            for arg in args:
                s = single_id(arg)
                if s is None:
                    continue
                for new in [x for x in args if x != arg]:
                    d = single_id(new)
                    if d is None:
                        continue
                    for f in cat["funcs"]:
                        wb, ws = f["answers"].get(arg), f["answers"].get(new)
                        if not wb or not ws:
                            continue
                        ids = lm.encode(f["template"].replace("{arg}", arg),
                                        max_length=128)
                        n += 1
                        g = greedy(ids, swap(lens, lm, layers=have, from_token=s,
                                             to_token=d, alpha=a_, mode="transfer"))
                        good += g == first_id(ws)
        out[f"4.1_a{a_}"] = good / max(n, 1)
        log(f"  4.1 flexgen a={a_}    {good}/{n}={good/max(n,1):.3f}")

    # ---- 5.1 causal
    passages = [(p["category"], p["text"]) for p in LANGS["passages"]]
    cats = sorted({c for c, _ in passages})
    for qn in ("explicit_q", "automatic_q"):
        q = LANGS["task"][qn]
        ch = n = 0
        for cat, text in passages:
            s = single_id(cat)
            d = next((single_id(a) for a in cats if a != cat and single_id(a)), None)
            if s is None or d is None:
                continue
            n += 1
            ids = lm.encode(chat(f"{text}\n\n{q}"), max_length=512)
            b = greedy(ids)
            g = greedy(ids, swap(lens, lm, layers=have, from_token=s, to_token=d,
                                 alpha=2.0, mode="transfer"))
            ch += g != b
        out[f"5.1_{qn}"] = ch / max(n, 1)
        log(f"  5.1 {qn:<12} {ch}/{n}={ch/max(n,1):.3f}")

    # ---- 5.3 ablation (position-adaptive top-k)
    def ablate(random_control=False, seed=0):
        gen = torch.Generator(device=DEV).manual_seed(seed)

        def edit(flat, layer):
            # `J` is deleted at the end of run_for, after every use of this
            # closure; ruff's end-of-scope analysis cannot see that ordering.
            Jl = J[layer]  # noqa: F821
            x = flat.to(DEV, torch.float32)
            B, S, _ = x.shape
            lg = lm.unembed(x @ Jl.T).float()
            top = lg.topk(K, dim=-1).indices
            rows = W[top.reshape(-1)].to(DEV, torch.float32) - Wm.to(DEV, torch.float32)
            D = (rows @ Jl).reshape(B, S, K, -1)
            if random_control:
                D = torch.randn(D.shape, generator=gen, device=DEV)
            D = D / D.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            Q, _ = torch.linalg.qr(D.transpose(-1, -2))
            c = torch.einsum("bsd,bsdr->bsr", x, Q)
            return (x - torch.einsum("bsr,bsdr->bsd", c, Q)).to(flat.dtype).to(flat.device)

        return Intervention(lm, have, edit)

    base_pred = []
    for p in CORPUS:
        ids = lm.encode(p, max_length=128)
        base_pred.append((ids, model(ids).logits[0].argmax(-1).cpu()))
    for nm, ctx in (("J-space", ablate()), ("random", ablate(random_control=True))):
        ag = tt = 0
        for ids, bp in base_pred:
            with ctx:
                pr = model(ids).logits[0].argmax(-1).cpu()
            ag += int((pr == bp).sum())
            tt += bp.numel()
        mh = 0
        for ids, _s, _d, a, _w, _base in prep:
            with ctx:
                mh += int(model(ids).logits[0, -1].argmax()) == a
        out[f"5.3_{nm}_agree"] = ag / tt
        out[f"5.3_{nm}_multihop"] = mh / len(prep)
        log(f"  5.3 {nm:<8} agree {ag/tt:.3f}   multi-hop {mh/len(prep):.3f}")

    # ---- 1.3 concept-vector decomposition
    by_cat = defaultdict(list)
    for it in PROBE:
        by_cat[it.get("category", "?")].append(it)
    cats2 = [c for c, v in by_cat.items() if len(v) >= 3][:6]
    sums = {c: {l: None for l in have} for c in cats2}
    cnt = {c: 0 for c in cats2}
    for c in cats2:
        for it in by_cat[c]:
            ids = lm.encode(it["prompt"], max_length=256)
            with ActivationRecorder(lm.layers, at=have) as rec:
                lm.forward(ids)
                for l in have:
                    h = rec.activations[l][0, -1].detach().to(DEV).float().flatten()
                    sums[c][l] = h if sums[c][l] is None else sums[c][l] + h
            cnt[c] += 1
    grand = {l: sum(sums[c][l] / cnt[c] for c in cats2) / len(cats2) for l in have}
    shares = []
    for c in cats2:
        for l in have:
            v = sums[c][l] / cnt[c] - grand[l]
            lg = lm.unembed(v.unsqueeze(0) @ J[l].T).float().reshape(-1)
            top = lg.topk(25).indices
            rows = W[top].to(DEV, torch.float32) - Wm.to(DEV, torch.float32)
            D = rows @ J[l]
            D = D / D.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            Q, _ = torch.linalg.qr(D.T)
            pr = Q @ (Q.T @ v)
            t = float(v.pow(2).sum())
            if t > 0:
                shares.append(float(pr.pow(2).sum()) / t)
    sh = sorted(shares)
    out["1.3_median"] = sh[len(sh) // 2] if sh else float("nan")
    log(f"  1.3 J-space share median {out['1.3_median']:.4f} (n={len(sh)})")

    # ---- 6.2 band statistics (top-1 accuracy per layer, sampled)
    acc = {l: [0, 0] for l in have}
    for p in CORPUS:
        ids = lm.encode(p, max_length=128)
        S = ids.shape[1]
        if S < SKIP + 4:
            continue
        nxt = ids[0, 1:].to(DEV)
        with ActivationRecorder(lm.layers, at=have) as rec:
            lm.forward(ids)
            a2 = {l: rec.activations[l][0].detach() for l in have}
        for l in have:
            H = a2[l].to(DEV).float()
            lg = lm.unembed(H.flatten(1) @ J[l].T).float()[SKIP:S - 1].to(DEV)
            gold = nxt[SKIP:S - 1]
            acc[l][0] += int((lg.argmax(-1) == gold).sum())
            acc[l][1] += lg.shape[0]
    for l in (have[0], have[len(have) // 2], have[-1]):
        v = acc[l][0] / max(acc[l][1], 1)
        out[f"6.2_top1_L{l}"] = v
        log(f"  6.2 top-1 acc L{l:<3} {v:.4f}")

    del J, lens
    torch.cuda.empty_cache()
    return out


# Build the disjoint-100 noise-floor lens if it is not on disk yet. The running
# chain calls this script directly after the n=1000 fit and never invokes a
# separate builder, so doing it here is what keeps the noise floor from silently
# going missing -- without it the comparison has no scale.
_DISJ = "/data3/fan-test/jlens_out/v4flash_disjoint100_s128.pt"
if not os.path.exists(_DISJ):
    try:
        full = torch.load("/data3/fan-test/jlens_out/band_n1000_ckpt.pt",
                          map_location="cpu", weights_only=False)
        snap = torch.load("/data3/fan-test/jlens_out/band_n900_snapshot.pt",
                          map_location="cpu", weights_only=False)
        d = full["n_done"] - snap["n_done"]
        if d <= 0:
            raise ValueError(f"no disjoint block: {full['n_done']} vs {snap['n_done']}")
        jac = {l: (full["jacobian_sum"][l] - snap["jacobian_sum"][l]) / d
               for l in full["source_layers"]}
        JacobianLens(jacobians=jac, n_prompts=d, d_model=full["d_target"],
                     d_source=full["d_source"]).save(_DISJ)
        log(f"built disjoint lens from {d} prompts ({snap['n_done']}-{full['n_done']})")
        del full, snap, jac
    except Exception as exc:
        log(f"could not build disjoint lens: {type(exc).__name__}: {exc}")

results = {}
for tag, path in LENSES:
    try:
        results[tag] = run_for(tag, path)
    except Exception as exc:
        import traceback
        log(f"!! {tag} FAILED {type(exc).__name__}: {exc}")
        traceback.print_exc()

log("=" * 70)
log("SIDE BY SIDE")
if "n=100" in results and "n=1000" in results:
    a, b = results["n=100"], results["n=1000"]
    d = results.get("disjoint100")
    log(f"  {'metric':<22} {'n=100':>9} {'n=1000':>9} {'delta':>9}")
    for k in a:
        if k in b:
            log(f"  {k:<22} {a[k]:>9.4f} {b[k]:>9.4f} {b[k]-a[k]:>+9.4f}")
    big = [k for k in a if k in b and abs(b[k] - a[k]) > 0.05]
    log(f"  metrics moving more than 0.05: {big if big else 'NONE'}")
    if d:
        log("")
        log(f"  {'metric':<22} {'|n1000-n100|':>13} {'|disj-n100|':>13} {'verdict':>12}")
        for k in a:
            if k in b and k in d:
                eff, noise = abs(b[k] - a[k]), abs(d[k] - a[k])
                v = "above noise" if eff > noise else "within noise"
                log(f"  {k:<22} {eff:>13.4f} {noise:>13.4f} {v:>12}")
        log("  an n=1000 effect only counts if it exceeds the disjoint-100 noise floor")
    log("  prediction on record was: no meaningful change")
json.dump(results, open(f"{SCRATCH}/exp_n1000_compare.json", "w"), indent=1)
log("DONE")
