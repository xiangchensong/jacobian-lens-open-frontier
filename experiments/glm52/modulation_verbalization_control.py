"""Verbalization control for GLM-5.2's white-bear result: re-run the suppress and
focus conditions, record whether the model wrote the concept word in its own
generated text, and report the hit rate restricted to trials where it did not.

This is the control behind the reported 0.917 suppress rate: GLM verbalized the
concept in 0 of 12 trials, so the lens is reading workspace content and not an
output token it can see. (The V4 comparison in the same table verbalized in 10 of
13 focus trials, which is why this check is reported per model.)

modulation.py gave suppress 0.917 vs an absent-concept floor of 0.000 -- a clean white
bear. But GLM is a reasoning model: if its generated thinking literally writes the
concept word ("I must not think about elephant"), the lens is reading an output
token, not the workspace. This re-runs suppress and prints (a) whether the concept
string appears in the generated text and (b) the hit rate restricted to trials where
it does NOT appear. Only the restricted rate is reportable.
"""
import json
import os
import time

import torch

S = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(S, exist_ok=True)

import jlens.fp8_autograd  # noqa: F401  must precede model load: registers fp8 backward


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
from transformers import AutoModelForCausalLM, AutoTokenizer

M = "/data3/fan-test/models/GLM-5.2-FP8"
n = torch.cuda.device_count()
model = AutoModelForCausalLM.from_pretrained(
    M, dtype="auto", device_map="auto", max_memory={i: "146GiB" for i in range(n)},
    attn_implementation="eager", experts_implementation="grouped_mm")
model.eval()
tok = AutoTokenizer.from_pretrained(M)
log("LOADED")
from jlens.hf import from_hf
from jlens.hooks import ActivationRecorder
from jlens.lens import JacobianLens

lm = from_hf(model, tok)
lens = JacobianLens.load("/data3/fan-test/jlens_out/glm52_band_n100_s128.pt")
BAND = lens.source_layers
J = {l: lens.jacobians[l].to("cuda:0") for l in BAND}
def chat(u): return tok.apply_chat_template([{"role": "user", "content": u}],
                                            tokenize=False, add_generation_prompt=True)
def sid(w):
    for v in (" " + w, w, " " + w.capitalize(), w.capitalize()):
        e = tok.encode(v, add_special_tokens=False)
        if len(e) == 1:
            return e[0]
CON = ["elephant","ocean","guitar","mountain","coffee","rocket","penguin",
       "castle","violin","desert","comet","orchid","harbor"]
NGEN = 8
res = {}
for mode, tpl in (("suppress", "Do NOT think about {c} at all. Name a color."),
                  ("focus", "Think carefully about {c} while answering: name a color.")):
    hit = n_all = hit_clean = n_clean = 0
    with torch.no_grad():
        for c in CON:
            t = sid(c)
            if t is None:
                continue
            ids = lm.encode(chat(tpl.format(c=c)), max_length=128)
            plen = ids.shape[1]
            for _ in range(NGEN):
                nx = int(model(ids).logits[0, -1].argmax())
                ids = torch.cat([ids, torch.tensor([[nx]], device=ids.device)], 1)
            gen_txt = tok.decode(ids[0, plen:].tolist()).lower()
            verbalized = c.lower()[:5] in gen_txt
            with ActivationRecorder(lm.layers, at=BAND) as rec:
                lm.forward(ids)
                acts = {l: rec.activations[l][0, plen:].detach() for l in BAND}
            got = False
            for l in BAND:
                lg = lm.unembed(acts[l].to("cuda:0").float() @ J[l].T).float()
                if int(((lg > lg[:, t:t+1]).sum(-1) == 0).sum()) > 0:
                    got = True
                    break
            hit += got
            n_all += 1
            if not verbalized:
                hit_clean += got
                n_clean += 1
            log(f"  {mode:<9} {c:<9} hit={int(got)} verbalized={int(verbalized)}  "
                f"gen={gen_txt[:48]!r}")
    res[mode] = dict(all=hit / max(n_all, 1), n_all=n_all,
                     clean=hit_clean / max(n_clean, 1), n_clean=n_clean)
    log(f"  == {mode}: all {hit}/{n_all}={res[mode]['all']:.3f} | "
        f"NOT-verbalized {hit_clean}/{n_clean}="
        f"{res[mode]['clean'] if n_clean else float('nan'):.3f}")
json.dump(res, open(f"{S}/glm_wb_check.json", "w"), indent=1)
log("VERDICT: only the NOT-verbalized rate is reportable as a workspace effect")
log("DONE")
