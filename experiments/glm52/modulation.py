"""Experiment 2.2 on GLM-5.2 -- directed modulation, measured at generated positions.

Both earlier attempts were artifacts -- last-position-only floored at 0, all-positions
ceilinged at 1 by reading the prompt tokens where the concept word literally appears.
The white-bear question is whether the concept remains active WHILE THE MODEL ANSWERS,
so: run the instruction, generate a few tokens, and read the lens at those generated
positions. The concept never appears literally there, so a hit means the workspace is
carrying it.

Control included: a concept never mentioned in the prompt at all ("absent"), which
bounds how often this readout fires by chance.
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
NGEN = 8

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
from jlens.hf import from_hf
from jlens.hooks import ActivationRecorder
from jlens.lens import JacobianLens

lm = from_hf(model, tok)
lens = JacobianLens.load(LENS)
BAND = lens.source_layers
J = {l: lens.jacobians[l].to(DEV) for l in BAND}
log(f"band L{BAND[0]}-{BAND[-1]}, reading {NGEN} generated positions")

def chat(u):
    return tok.apply_chat_template([{"role": "user", "content": u}],
                                   tokenize=False, add_generation_prompt=True)

def single_id(w):
    for v in (" " + w, w, " " + w.capitalize(), w.capitalize()):
        e = tok.encode(v, add_special_tokens=False)
        if len(e) == 1:
            return e[0]

CONCEPTS = ["elephant", "ocean", "guitar", "mountain", "coffee", "rocket",
            "penguin", "castle", "violin", "desert", "comet", "orchid", "harbor"]
MODES = {"focus": "Think carefully about {c} while answering: name a color.",
         "mention": "Name a color. (Unrelated: {c} exists.)",
         "suppress": "Do NOT think about {c} at all. Name a color.",
         "absent": "Name a color."}

@torch.no_grad()
def run():
    res = {}
    for m, tpl in MODES.items():
        hit = n = 0
        for c in CONCEPTS:
            t = single_id(c)
            if t is None:
                continue
            text = tpl.format(c=c) if "{c}" in tpl else tpl
            ids = lm.encode(chat(text), max_length=128)
            plen = ids.shape[1]
            for _ in range(NGEN):          # generate, then read ONLY these positions
                nx = int(model(ids).logits[0, -1].argmax())
                ids = torch.cat([ids, torch.tensor([[nx]], device=ids.device)], 1)
            with ActivationRecorder(lm.layers, at=BAND) as rec:
                lm.forward(ids)
                acts = {l: rec.activations[l][0, plen:].detach() for l in BAND}
            got = False
            for l in BAND:
                H = acts[l].to(DEV).float()
                lg = lm.unembed(H @ J[l].T).float()
                if int(((lg > lg[:, t:t + 1]).sum(-1) == 0).sum()) > 0:
                    got = True
                    break
            hit += got
            n += 1
        res[m] = hit / max(n, 1)
        log(f"  {m:<9} {hit}/{n} = {res[m]:.3f}")
    log("  [V4: focus 0.383 > mention 0.167 > suppress 0.000 (no white bear)]")
    base = res.get("absent", 0.0)
    log(f"  chance floor (concept never mentioned): {base:.3f}")
    if max(res[k] for k in ("focus", "mention", "suppress")) <= base:
        log("  ==> NO SIGNAL above the absent-concept control; uninformative")
    else:
        wb = res["suppress"] > base
        log(f"  ==> ordering focus {res['focus']:.3f} / mention {res['mention']:.3f} / "
            f"suppress {res['suppress']:.3f} vs floor {base:.3f}")
        log(f"  ==> white bear: {'YES (suppress above floor)' if wb else 'NO (suppress at floor)'}")
    json.dump(res, open(f"{S}/glm_battery4.json", "w"), indent=1)

try:
    run()
except Exception as exc:
    import traceback
    log(f"!! FAILED {type(exc).__name__}: {exc}")
    traceback.print_exc()
log("DONE")
