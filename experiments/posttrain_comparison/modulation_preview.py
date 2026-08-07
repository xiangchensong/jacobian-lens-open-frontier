"""Experiment 2.2 -- directed modulation (white bear), measured at generated
positions, on the preview checkpoint. Path-only derivation of
deepseek_v4/modulation.py.

GLM shows suppress 0.917 (verbalization-controlled); V4 was recorded at 0.000 -- but
under a DIFFERENT protocol. 2.2 broke twice on GLM purely from protocol choice, so a
cross-family claim needs both models measured the same way. This re-runs V4 with the
generated-positions readout, the absent-concept floor, and the verbalization control.

Does the white-bear result survive excluding literal verbalization?

an earlier pass gave suppress 0.917 vs an absent-concept floor of 0.000 -- a clean white
bear. But GLM is a reasoning model: if its generated thinking literally writes the
token, not the workspace. This re-runs suppress and prints (a) whether the concept
string appears in the generated text and (b) the hit rate restricted to trials where
it does NOT appear. Only the restricted rate is reportable.
"""
# NOTE: paths under /data3 (model weights, fitted lenses in /data3/fan-test/jlens_out)
# are machine-specific to the original 8xB200 host; adjust for your environment.
import json
import os
import sys
import time

import torch

S = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(S, exist_ok=True)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


from transformers import AutoModelForCausalLM, AutoTokenizer, FineGrainedFP8Config

M = "/data3/fan-test/models/DeepSeek-V4-Flash"
n = torch.cuda.device_count()
model = AutoModelForCausalLM.from_pretrained(
    M, dtype="auto", device_map="auto", max_memory={i: "146GiB" for i in range(n)},
    attn_implementation="eager", experts_implementation="grouped_mm",
    quantization_config=FineGrainedFP8Config(dequantize=True))
model.eval()
tok = AutoTokenizer.from_pretrained(M)
log("LOADED")
from jlens.hf import from_hf
from jlens.hooks import ActivationRecorder
from jlens.lens import JacobianLens

lm = from_hf(model, tok)
lens = JacobianLens.load("/data3/fan-test/jlens_out/v4preview_band_n100_s128.pt")
BAND = lens.source_layers
J = {l: lens.jacobians[l].to("cuda:0") for l in BAND}
sys.path.insert(0, "/data3/fan-test/models/DeepSeek-V4-Flash/encoding")
import encoding_dsv4 as DSV4


def chat(u):
    return DSV4.encode_messages([{"role": "user", "content": u}], thinking_mode="chat")
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
                h = acts[l].to("cuda:0").float().flatten(start_dim=1)   # [pos,4,4096]->[pos,16384]
                lg = lm.unembed(h @ J[l].T, collapse=False).float()
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
json.dump(res, open(f"{S}/v4_wb_check.json", "w"), indent=1)
log("VERDICT: only the NOT-verbalized rate is reportable as a workspace effect")
log("DONE")
