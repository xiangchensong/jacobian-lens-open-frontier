"""GLM-5.2 experiments 4.1 (broadcast, generated-answer criterion), 2.2, 5.4 and
5.2 -- the run whose 4.1 and 5.4 numbers are the ones reported.

This supersedes the 4.1 and 2.2 in interventions.py, whose scoring criteria the
write-up retracts:
  4.1  interventions.py scored "swapped concept readable in band", which the swap
       guarantees by construction -- circular, and it is why the 0.750 it produced
       is not reported. Scored here as whether the model's GENERATED answer to a
       different downstream task changes to the swapped concept, giving 0.375.
  2.2  interventions.py scanned only the last position, so all three conditions
       read 0/12 -- an uninformative null it printed as "no white bear". Scanned
       over all positions here. (The reported 2.2 protocol is later again:
       modulation.py reads only generated positions, with an absent-concept
       floor.)
  5.4  is GLM's continuation of the load-bearing ablation: experiential-language
       ablation with a matched-rank random control, the run in which GLM flips to
       "I'm an AI ... without feelings or a real 'self'".

  5.2  readout selectivity, deliberate vs automatic content in the lens. NOTE:
       as written this item is confounded -- the deliberate set is
       arithmetic-dominated and arithmetic is separately known to be
       Jacobian-invisible on these models -- so its output is not reported.
"""
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

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

from transformers import AutoModelForCausalLM, AutoTokenizer

n_gpu = torch.cuda.device_count()
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype="auto", device_map="auto",
    max_memory={i: "146GiB" for i in range(n_gpu)},
    attn_implementation="eager", experts_implementation="grouped_mm")
model.eval()
tok = AutoTokenizer.from_pretrained(MODEL)
log("LOADED")

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
log(f"band L{BAND[0]}-{BAND[-1]}")


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


@torch.no_grad()
def generate(text, ctx=None, n=40):
    ids = lm.encode(text, max_length=512)
    out = []
    for _ in range(n):
        if ctx is None:
            nx = int(model(ids).logits[0, -1].argmax())
        else:
            with ctx:
                nx = int(model(ids).logits[0, -1].argmax())
        out.append(nx)
        ids = torch.cat([ids, torch.tensor([[nx]], device=ids.device)], 1)
    return tok.decode(out)


# ==================================================== 2.2 FIXED: all positions
@torch.no_grad()
def exp22_fixed():
    log("=" * 72)
    log("(2.2 FIXED) modulation, scanning ALL positions [b2 read last pos only = null]")
    concepts = ["elephant", "ocean", "guitar", "mountain", "coffee", "rocket",
                "penguin", "castle", "violin", "desert", "comet", "orchid", "harbor"]
    modes = {"focus": "Think carefully about {c} while answering: name a color.",
             "mention": "Name a color. (Unrelated: {c} exists.)",
             "suppress": "Do NOT think about {c} at all. Name a color."}
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
                acts = {l: rec.activations[l][0].detach() for l in BAND}
            got = False
            for l in BAND:
                H = acts[l].to(DEV).float()
                lg = lm.unembed(H @ J[l].T).float()          # [pos, vocab]
                if int(((lg > lg[:, t:t+1]).sum(-1) == 0).sum()) > 0:
                    got = True
                    break
            hit += got
            n += 1
        res[m] = hit / max(n, 1)
        log(f"  {m:<9} {hit}/{n} = {res[m]:.3f}")
    log("  [V4: focus 0.383 > mention 0.167 > suppress 0.000]")
    if max(res.values()) == 0:
        log("  ==> STILL NULL: probe finds nothing in any condition; uninformative")
    else:
        log(f"  ==> ordering {'reproduces' if res['focus'] >= res['mention'] >= res['suppress'] else 'DIFFERS'}"
            f"; white bear {'YES' if res['suppress'] > 0 else 'NO'}")
    OUT["22_fixed"] = res


# ==================================================== 4.1 FIXED: real downstream
@torch.no_grad()
def exp41_fixed():
    log("=" * 72)
    log("(4.1 FIXED) broadcast: does the GENERATED downstream answer change?")
    items = json.load(open("data/experiments/probe-swap.json"))["items"][:40]
    tasks = {"translate": "\nNow translate that answer into French. Answer:",
             "rhyme": "\nNow give a word that rhymes with that answer. Answer:"}
    hit = n = unchanged = 0
    for it in items:
        s, d = single_id(it["intermediate"]), single_id(it["swap_to"])
        if None in (s, d):
            continue
        to_txt = it["swap_to"].lower()
        for _tname, suffix in tasks.items():
            p = it["prompt"] + suffix
            baseline = generate(p, None, n=12).lower()
            ctx = swap(lens, lm, layers=BAND, from_token=s, to_token=d,
                       alpha=2.0, mode="transfer")
            got = generate(p, ctx, n=12).lower()
            n += 1
            if to_txt[:5] in got and to_txt[:5] not in baseline:
                hit += 1
            elif got.strip() == baseline.strip():
                unchanged += 1
    log(f"  downstream answer switched to swapped concept: {hit}/{n} = {hit/max(n,1):.3f}")
    log(f"  unchanged (edit too weak): {unchanged}/{n}   [V4 0.208 ; paper 0.526]")
    OUT["41_fixed"] = dict(hit=hit, n=n, rate=hit / max(n, 1), unchanged=unchanged)


# ==================================================== 5.4 experiential
@torch.no_grad()
def exp54():
    log("=" * 72)
    log("(5.4) experiential-language ablation + matched-rank random control")
    prompt = chat("Describe what it is like for you, right now, in one short "
                  "paragraph. Write in the first person.")
    conds = {"unablated": None,
             f"J-space L{THIRD[0]}-{THIRD[-1]}": adaptive(THIRD),
             f"J-space L{BAND[0]}-{BAND[-1]}": adaptive(BAND),
             f"random L{BAND[0]}-{BAND[-1]}": adaptive(BAND, random_control=True)}
    outs = {}
    for name, ctx in conds.items():
        txt = generate(prompt, ctx, n=45).replace("\n", " ")[:150]
        outs[name] = txt
        log(f"  {name:<22} {txt!r}")
    log("  [V4: full-band ablation flipped to 'I am a language model, and I do not have a...']")
    OUT["54"] = outs


# ==================================================== 5.2 readout selectivity
@torch.no_grad()
def exp52():
    log("=" * 72)
    log("(5.2) readout: deliberate vs automatic content in the lens")
    pairs = [("What is 17 times 3?", "51", "deliberate"),
             ("The capital of France is", "Paris", "automatic"),
             ("What is 128 divided by 4?", "32", "deliberate"),
             ("The opposite of hot is", "cold", "automatic"),
             ("If all cats are animals and Rex is a cat, Rex is a", "animal", "deliberate"),
             ("Water freezes at zero degrees", "Celsius", "automatic")]
    res = {"deliberate": [], "automatic": []}
    for q, ans, kind in pairs:
        t = first_id(ans)
        if t is None:
            continue
        ids = lm.encode(chat(q) if kind == "deliberate" else q, max_length=128)
        with ActivationRecorder(lm.layers, at=BAND) as rec:
            lm.forward(ids)
            acts = {l: rec.activations[l][0, -1].detach() for l in BAND}
        best = min(int((lm.unembed(acts[l].to(DEV).float().unsqueeze(0) @ J[l].T)
                        .reshape(-1) > lm.unembed(acts[l].to(DEV).float().unsqueeze(0)
                        @ J[l].T).reshape(-1)[t]).sum()) for l in BAND)
        res[kind].append(best)
        log(f"  [{kind:<10}] {q[:38]:<38} -> {ans!r} band-min rank {best}")
    for k, v in res.items():
        if v:
            log(f"  {k}: median band-min rank {sorted(v)[len(v)//2]}")
    OUT["52"] = res


for fn in (exp22_fixed, exp41_fixed, exp54, exp52):
    try:
        fn()
    except Exception as exc:
        import traceback
        log(f"!! {fn.__name__} FAILED {type(exc).__name__}: {exc}")
        traceback.print_exc()
json.dump(OUT, open(f"{S}/glm_battery3.json", "w"), indent=1, default=str)
log("DONE")
