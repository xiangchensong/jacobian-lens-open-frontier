"""GLM-5.2 causal experiments: 5.3 ablation with control, 2.2, 3.5, 4.1, 1.3.

5.3  Is SELECTIVITY (reasoning collapses, automatic prediction spared) universal,
     or something frontier training adds? Claude: spared >0.90. V4: 0.508. GLM decides.
2.2  Is the "white bear" residual Claude-specific? V4: 0 of 156. GLM decides.
3.5  Ordered resolution, and whether the clearing effect seen on V4 (consumed
     intermediates pushed below chance) is cross-family. Few-shot format forces
     inline answers so the readout position is well defined.
4.1  Does BROADCAST rate track capability like swap did (0.455 vs 0.434)?
1.3  Is V4's extreme J-space compression (0.4% vs paper 6-15%) a family fact?
5.2  Deliberate/automatic readout asymmetry. NOTE: as written this item is
     confounded -- the deliberate set is arithmetic-dominated, and arithmetic is
     separately known to be Jacobian-invisible on these models, so it cannot
     separate deliberate-vs-automatic from arithmetic-vs-non-arithmetic. Its
     output is not reported.

Chat encoding uses GLM's shipped template (no encoding_dsv4). Square lens path
throughout: no collapse=, no target_module, TARGET = n_layers - 1.
"""
# NOTE: paths under /data3 (model weights, fitted lenses in /data3/fan-test/jlens_out)
# are machine-specific to the original 8xB200 host; adjust for your environment.
import json
import os
import time

import torch

S = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(S, exist_ok=True)

import jlens.fp8_autograd  # noqa: F401  must precede model load: fp8 backward formulas

MODEL = "/data3/fan-test/models/GLM-5.2-FP8"
LENS = "/data3/fan-test/jlens_out/glm52_band_n100_s128.pt"
DEV = "cuda:0"
K = 10
OUT = {}

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
from jlens.intervene import Intervention, swap
from jlens.lens import JacobianLens

lm = from_hf(model, tok)
lens = JacobianLens.load(LENS)
BAND = lens.source_layers
THIRD = BAND[:len(BAND) // 3]
J = {l: lens.jacobians[l].to(DEV) for l in BAND}
W = model.lm_head.weight
Wm = W.mean(0, keepdim=True)
log(f"band L{BAND[0]}-{BAND[-1]}  K={K}")


def chat(user, prefill=""):
    s = tok.apply_chat_template([{"role": "user", "content": user}],
                                tokenize=False, add_generation_prompt=True)
    return s + prefill


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


def adaptive(layers, random_control=False, seed=0):
    """Position-adaptive top-K ablation -- the protocol that worked on V4."""
    gen = torch.Generator(device=DEV).manual_seed(seed)
    def edit(flat, layer):
        Jl = J[layer]
        x = flat.to(DEV, torch.float32)
        B, Sq, _ = x.shape
        lg = lm.unembed(x @ Jl.T).float()
        top = lg.topk(K, dim=-1).indices
        rows = W[top.reshape(-1)].to(DEV, torch.float32) - Wm.to(DEV, torch.float32)
        D = (rows @ Jl).reshape(B, Sq, K, -1)
        if random_control:
            D = torch.randn(D.shape, generator=gen, device=DEV)
        D = D / D.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        Q, _ = torch.linalg.qr(D.transpose(-1, -2))
        c = torch.einsum("bsd,bsdr->bsr", x, Q)
        return (x - torch.einsum("bsr,bsdr->bsd", c, Q)).to(flat.dtype).to(flat.device)
    return Intervention(lm, layers, edit)


# ============================================================ 5.3 THE ARBITER
@torch.no_grad()
def exp53():
    log("=" * 72)
    log("(5.3) ablation: does GLM show SELECTIVITY? [Claude >0.90 spared, V4 0.508]")
    items = json.load(open("data/experiments/probe-swap.json"))["items"]
    prep = []
    for it in items:
        a = first_id(it["answer"])
        if a is None:
            continue
        ids = lm.encode(it["prompt"], max_length=256)
        prep.append((ids, a))
    def multihop(ctx):
        n = 0
        for ids, a in prep:
            if ctx is None:
                g = int(model(ids).logits[0, -1].argmax())
            else:
                with ctx:
                    g = int(model(ids).logits[0, -1].argmax())
            n += g == a
        return n / len(prep)
    base = multihop(None)
    abl = multihop(adaptive(BAND))
    rnd = multihop(adaptive(BAND, random_control=True))
    log(f"  multihop  unablated {base:.3f}  J-ablated {abl:.3f}  random {rnd:.3f}")
    log("            [V4: 0.589 / 0.100 / 0.589]")

    # automatic next-token prediction on pretraining text
    prompts = ex.load_wikitext_prompts(20, min_chars=600)
    def agreement(ctx):
        hit = tot = 0
        for p in prompts:
            ids = lm.encode(p, max_length=128)
            ref = model(ids).logits[0, 16:-1].argmax(-1)
            if ctx is None:
                got = ref
            else:
                with ctx:
                    got = model(ids).logits[0, 16:-1].argmax(-1)
            hit += int((got == ref).sum())
            tot += ref.numel()
        return hit / tot
    a_abl = agreement(adaptive(BAND))
    a_rnd = agreement(adaptive(BAND, random_control=True))
    log(f"  pretrain top-1 agreement  J-ablated {a_abl:.3f}  random {a_rnd:.3f}")
    log("            [Claude >0.90 = SELECTIVE ; V4 0.508 = NOT selective]")
    verdict = ("SELECTIVE (like Claude)" if a_abl > 0.85 else
               "NOT selective (like V4)" if a_abl < 0.70 else "INTERMEDIATE")
    log(f"  ==> GLM verdict: {verdict}")
    OUT["53"] = dict(base=base, abl=abl, rnd=rnd, agree_abl=a_abl,
                     agree_rnd=a_rnd, verdict=verdict)


# ============================================================ 2.2 white bear
@torch.no_grad()
def exp22():
    log("=" * 72)
    log("(2.2) directed modulation: is the 'white bear' residual Claude-specific?")
    concepts = ["elephant", "ocean", "guitar", "mountain", "coffee", "rocket",
                "penguin", "castle", "violin", "desert", "comet", "orchid",
                "harbor"]
    modes = {
        "focus":    "Think carefully about {c} while answering: name a color.",
        "mention":  "Name a color. (Unrelated: {c} exists.)",
        "suppress": "Do NOT think about {c} at all. Name a color.",
    }
    res = {}
    for m, tpl in modes.items():
        hit = n = 0
        for c in concepts:
            t = single_id(c)
            if t is None:
                continue
            ids = lm.encode(chat(tpl.format(c=c)), max_length=128)
            with ActivationRecorder(lm.layers, at=BAND) as rec:
                lm.forward(ids)
                acts = {l: rec.activations[l][0, -1].detach() for l in BAND}
            got = False
            for l in BAND:
                lg = lm.unembed(acts[l].to(DEV).float().unsqueeze(0) @ J[l].T)
                if int((lg.reshape(-1) > lg.reshape(-1)[t]).sum()) == 0:
                    got = True
                    break
            hit += got
            n += 1
        res[m] = hit / max(n, 1)
        log(f"  {m:<9} {hit}/{n} = {res[m]:.3f}")
    log("  [V4: focus 0.383 > mention 0.167 > suppress 0.000 (no white bear)]")
    log(f"  ==> white bear on GLM: {'YES' if res['suppress'] > 0 else 'NO'}")
    OUT["22"] = res


# ============================================================ 3.5 few-shot
@torch.no_grad()
def exp35f():
    log("=" * 72)
    log("(3.5f) few-shot arithmetic: ordered resolution + CLEARING effect")
    p = ("calc: 3+4=7\ncalc: 10*2=20\ncalc: 8-3=5\ncalc: (4+17)*2+7=")
    ids = lm.encode(p, max_length=64)
    with torch.no_grad():
        g = []
        cur = ids
        for _ in range(4):
            nx = int(model(cur).logits[0, -1].argmax())
            g.append(nx)
            cur = torch.cat([cur, torch.tensor([[nx]], device=cur.device)], 1)
    log(f"  greedy answer: {tok.decode(g)!r} (correct = 49)")
    with ActivationRecorder(lm.layers, at=BAND) as rec:
        lm.forward(ids)
        acts = {l: rec.activations[l][0, -1].detach() for l in BAND}
    tgt = {k: single_id(k) for k in ("21", "42", "49")}
    traj = {k: [] for k in tgt}
    first = {k: None for k in tgt}
    for l in BAND:
        lg = lm.unembed(acts[l].to(DEV).float().unsqueeze(0) @ J[l].T).float().reshape(-1)
        for k, t in tgt.items():
            r = int((lg > lg[t]).sum()) if t is not None else -1
            traj[k].append(r)
            if r == 0 and first[k] is None:
                first[k] = l
    log(f"  first rank-0: {first}   [V4: 21@L27 -> 42@L30 -> 49@L36, in order]")
    V = W.shape[0]
    for k in tgt:
        mx = max(traj[k])
        log(f"  {k}: best rank {min(traj[k])}, worst {mx} "
            f"({'BELOW' if mx > V/2 else 'above'} chance {V//2}) -> "
            f"{'clearing' if mx > V/2 else 'no clearing'}")
    OUT["35f"] = dict(first=first, traj=traj, greedy=tok.decode(g))


# ============================================================ 4.1 broadcast
@torch.no_grad()
def exp41():
    log("=" * 72)
    log("(4.1) flexible generalization: does BROADCAST track capability too?")
    items = json.load(open("data/experiments/probe-swap.json"))["items"][:60]
    tasks = {"translate": "Translate the last answer into French: ",
             "rhyme": "Give a word that rhymes with the last answer: "}
    hit = n = 0
    for it in items:
        s, d = single_id(it["intermediate"]), single_id(it["swap_to"])
        if None in (s, d):
            continue
        for _tname, suffix in tasks.items():
            ids = lm.encode(it["prompt"] + "\n" + suffix, max_length=256)
            ctx = swap(lens, lm, layers=BAND, from_token=s, to_token=d,
                       alpha=2.0, mode="transfer")
            with ctx:
                with ActivationRecorder(lm.layers, at=BAND) as rec:
                    lm.forward(ids)
                    acts = {l: rec.activations[l][0, -1].detach() for l in BAND}
            got = False
            for l in BAND:
                lg = lm.unembed(acts[l].to(DEV).float().unsqueeze(0) @ J[l].T)
                if int((lg.reshape(-1) > lg.reshape(-1)[d]).sum()) < 5:
                    got = True
                    break
            hit += got
            n += 1
    log(f"  swapped concept reaches rank<5 downstream: {hit}/{n} = {hit/max(n,1):.3f}")
    log("  [V4 0.208 ; paper 0.526]")
    OUT["41"] = hit / max(n, 1)


# ============================================================ 1.3 variance
@torch.no_grad()
def exp13():
    log("=" * 72)
    log("(1.3) J-space variance share  [V4 0.4% ; paper 6-15%]")
    prompts = ex.load_wikitext_prompts(10, min_chars=600)
    shares = []
    for l in BAND[::6]:
        num = den = 0.0
        for p in prompts:
            ids = lm.encode(p, max_length=128)
            with ActivationRecorder(lm.layers, at=[l]) as rec:
                lm.forward(ids)
                h = rec.activations[l][0, 16:-1].detach().to(DEV).float()
            lg = lm.unembed(h @ J[l].T).float()
            top = lg.topk(K, dim=-1).indices
            rows = W[top.reshape(-1)].to(DEV, torch.float32) - Wm.to(DEV, torch.float32)
            D = (rows @ J[l]).reshape(h.shape[0], K, -1)
            D = D / D.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            Q, _ = torch.linalg.qr(D.transpose(-1, -2))
            c = torch.einsum("sd,sdr->sr", h, Q)
            num += float((c ** 2).sum())
            den += float((h ** 2).sum())
        shares.append((l, num / max(den, 1e-9)))
        log(f"  L{l:<3} variance in top-{K} J-space: {shares[-1][1]*100:.2f}%")
    OUT["13"] = shares


for fn in (exp53, exp22, exp35f, exp41, exp13):
    try:
        fn()
    except Exception as exc:
        import traceback
        log(f"!! {fn.__name__} FAILED {type(exc).__name__}: {exc}")
        traceback.print_exc()
json.dump(OUT, open(f"{S}/glm_battery2.json", "w"), indent=1, default=str)
log("DONE")
