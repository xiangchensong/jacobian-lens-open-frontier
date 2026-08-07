"""Fit a band lens (L19-39, S=128, n=100, per-prompt normalization) on
DeepSeek-V4-Flash-Base -- the lens section8_posttraining.py reads the base model
through.

Section 8's claim is that post-training installs the Assistant point of view in
J-space. Testing it needs a lens on the BASE model fitted identically to the
post-trained one -- same band, same corpus, same normalisation -- so any
difference in J-space contents is attributable to post-training and not to the
fit. n=100 is used rather than 1000: the n-scaling curve was flat from n=60, and
the n=1000 run confirmed 3.3 moves 0.0000, so a 25-hour fit would buy nothing.
"""
import time

import torch

MODEL="/data3/fan-test/models/DeepSeek-V4-Flash-Base"
OUT="/data3/fan-test/jlens_out"
SOURCE=list(range(19,40))
DIM_BATCH=16
MAX_SEQ=128
N=100


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


from transformers import AutoModelForCausalLM, AutoTokenizer, FineGrainedFP8Config

n_gpu=torch.cuda.device_count()
t0=time.time()
model=AutoModelForCausalLM.from_pretrained(MODEL, dtype="auto", device_map="auto",
    max_memory={i:"146GiB" for i in range(n_gpu)}, attn_implementation="eager",
    experts_implementation="grouped_mm",
    quantization_config=FineGrainedFP8Config(dequantize=True))
model.eval()
tok=AutoTokenizer.from_pretrained(MODEL)
log(f"BASE LOADED in {time.time()-t0:.0f}s")
import jlens.examples as ex
from jlens._logging import configure_logging
from jlens.fitting import fit
from jlens.hf import from_hf

configure_logging()
lm=from_hf(model,tok)
log(f"n_layers={lm.n_layers} d_model={lm.d_model} "
    f"d_source={getattr(lm,'d_source',lm.d_model)}")
prompts=ex.load_wikitext_prompts(N, min_chars=600)
t0=time.time()
lens=fit(lm, prompts, source_layers=SOURCE, dim_batch=DIM_BATCH,
         max_seq_len=MAX_SEQ, checkpoint_path=f"{OUT}/base_n100_ckpt.pt",
         checkpoint_every=25, resume=True, normalize_per_prompt=True)
log(f"FIT DONE in {time.time()-t0:.0f}s ({(time.time()-t0)/max(lens.n_prompts,1):.0f}s/prompt)")
p=f"{OUT}/v4base_band_n{lens.n_prompts}_s{MAX_SEQ}.pt"
lens.save(p)
log(f"saved {p}")
log("DONE")
