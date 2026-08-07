"""Fit the preview checkpoint's band lens, matched to 0731's recipe exactly.

Same corpus (WikiText-103, >=600 chars, first 100 records in stream order),
same band L19-39, same S=128, same skip_first=16, same per-prompt
normalization, same dim_batch. Any of these differing would make the
preview-vs-0731 comparison measure our settings instead of their training.
"""
# NOTE: paths under /data3 (model weights, fitted lenses in /data3/fan-test/jlens_out)
# are machine-specific to the original 8xB200 host; adjust for your environment.
import math
import time

import torch

MODEL = "/data3/fan-test/models/DeepSeek-V4-Flash"
OUT   = "/data3/fan-test/jlens_out/v4preview_band_n100_s128.pt"
BAND, DIM_BATCH, MAX_SEQ, N = list(range(19, 40)), 16, 128, 100
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
log("LOADED")
import jlens.examples as ex
from jlens._logging import configure_logging
from jlens.fitting import fit
from jlens.hf import from_hf
from jlens.protocol import d_source_of, d_target_of

configure_logging()
lm = from_hf(model, tok)
log(f"d_source={d_source_of(lm)} d_target={d_target_of(lm)} band L{BAND[0]}-{BAND[-1]} "
    f"{math.ceil(d_target_of(lm)/DIM_BATCH)} backwards/prompt")
prompts = ex.load_wikitext_prompts(N, min_chars=600)
t0 = time.time()
# fit() resolves the mHC target module itself via target_module_of(); it takes
# neither target_layer nor extra_modules. Same call shape as the n=1000 fit.
lens = fit(lm, prompts, source_layers=BAND,
           dim_batch=DIM_BATCH, max_seq_len=MAX_SEQ,
           checkpoint_path="/data3/fan-test/jlens_out/v4preview_ckpt.pt",
           checkpoint_every=10, resume=True, normalize_per_prompt=True)
lens.save(OUT)
log(f"FIT DONE {lens.n_prompts} prompts in {(time.time()-t0)/3600:.2f}h -> {OUT}")
log("DONE")
