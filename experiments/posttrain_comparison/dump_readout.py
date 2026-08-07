"""Dump each checkpoint's own workspace content on a fixed prompt set.

Usage: dump_readout.py {preview|cur|cur_disj}

Each variant loads its OWN model and reads it with its OWN lens -- that is what
"what is in this model's workspace" means. Comparing the dumps answers what
post-training changed; comparing two lenses on one model's activations would
answer the different question of how the readout maps differ.

`cur_disj` reads 0731 with the disjoint-corpus 0731 lens: same model, same
prompts, a lens fitted on different text. That is the floor -- whatever
difference it shows is corpus sampling, not training.

Writes top-k token ids per (layer, position) so the comparison is offline and
needs no second load.
"""
# NOTE: paths under /data3 (model weights, fitted lenses in /data3/fan-test/jlens_out)
# are machine-specific to the original 8xB200 host; adjust for your environment.
import os
import sys
import time

import torch

WHICH = sys.argv[1]
CFG = {"preview":  ("/data3/fan-test/models/DeepSeek-V4-Flash",
                    "/data3/fan-test/jlens_out/v4preview_band_n100_s128.pt"),
       "cur":      ("/data3/fan-test/models/DeepSeek-V4-Flash-0731",
                    "/data3/fan-test/jlens_out/v4flash_runAnorm_n100_s128.pt"),
       "cur_disj": ("/data3/fan-test/models/DeepSeek-V4-Flash-0731",
                    "/data3/fan-test/jlens_out/v4flash_disjoint100_s128.pt")}
MODEL, LENS = CFG[WHICH]
OUTD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUTD, exist_ok=True)
DEV, TOPK, NPROMPT = "cuda:0", 50, 20
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
log(f"LOADED {WHICH}")
import jlens.examples as ex
from jlens.hf import from_hf
from jlens.hooks import ActivationRecorder
from jlens.lens import JacobianLens

lm = from_hf(model, tok)
lens = JacobianLens.load(LENS)
BAND = sorted(lens.jacobians)
prompts = ex.load_wikitext_prompts(NPROMPT, min_chars=600)   # deterministic stream
out = {}
with torch.no_grad():
    for l in BAND:
        Jl = lens.jacobians[l].to(DEV, torch.float32)
        tops = []
        for p in prompts:
            ids = lm.encode(p, max_length=128)
            with ActivationRecorder(lm.layers, at=[l]) as rec:
                lm.forward(ids)
                h = rec.activations[l][0, 16:-1].detach().to(DEV, torch.float32)
            lg = lm.unembed(h.flatten(start_dim=-2) @ Jl.T, collapse=False).float()
            tops.append(lg.topk(TOPK, dim=-1).indices.cpu())
        out[l] = torch.cat(tops).to(torch.int32)
        del Jl
        torch.cuda.empty_cache()
        if l % 6 == BAND[0] % 6:
            log(f"  L{l} done, {out[l].shape[0]} positions")
torch.save(out, f"{OUTD}/readout_{WHICH}.pt")
log(f"saved readout_{WHICH}.pt  ({len(BAND)} layers)")
log("DONE")
