"""Fit the GLM-5.2 band lens (L34-71, n=100).

Band L34-71 is the PROJECTED band (44-91% of 78 layers, carried over from
DeepSeek's measured relative depth). It is a bootstrap assumption: the fitted
lens itself is what lets us run 6.2 and measure GLM's true band, after which the
band is corrected if needed. Fitting a wider slice up front would double cost
for layers the experiments may never read.

Settings mirror the DeepSeek recipe exactly where architecture allows:
S=128, skip_first=16, normalize_per_prompt=True, WikiText n=100.
dim_batch=32 per the measured probe (9.0 min/prompt, peak 110 GiB).
fp8 backward via the registered formulas in jlens.fp8_autograd (straight-through
w.r.t. the kernels' internal activation quantization -- see methods appendix).
"""
# NOTE: paths under /data3 (model weights, fitted lenses in /data3/fan-test/jlens_out)
# are machine-specific to the original 8xB200 host; adjust for your environment.
import time

import torch

import jlens.fp8_autograd  # noqa: F401  registers fp8 backward formulas

MODEL = "/data3/fan-test/models/GLM-5.2-FP8"
OUT = "/data3/fan-test/jlens_out"
SOURCE = list(range(34, 72))
DIM_BATCH = 32
MAX_SEQ = 128
N = 100

def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

from transformers import AutoModelForCausalLM, AutoTokenizer

n_gpu = torch.cuda.device_count()
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype="auto", device_map="auto",
    max_memory={i: "146GiB" for i in range(n_gpu)},
    attn_implementation="eager", experts_implementation="grouped_mm")
model.eval()
tok = AutoTokenizer.from_pretrained(MODEL)
log(f"LOADED in {time.time()-t0:.0f}s")

import jlens.examples as ex
from jlens._logging import configure_logging
from jlens.fitting import fit
from jlens.hf import from_hf

configure_logging()          # per-prompt progress lines; a silent 15h run is undebuggable

lm = from_hf(model, tok)
log(f"{type(lm).__name__} n_layers={lm.n_layers} d_model={lm.d_model}")
prompts = ex.load_wikitext_prompts(N, min_chars=600)
log(f"corpus: {len(prompts)} prompts; band L{SOURCE[0]}-{SOURCE[-1]}; "
    f"dim_batch={DIM_BATCH}")

# per-layer J is 6144^2 * 4 B = 151 MB; 38 layers -> 5.7 GB/checkpoint.
# checkpoint_every=10 keeps checkpoint I/O ~1% of runtime.
t0 = time.time()
lens = fit(lm, prompts, source_layers=SOURCE, dim_batch=DIM_BATCH,
           max_seq_len=MAX_SEQ, checkpoint_path=f"{OUT}/glm_band_n100_ckpt.pt",
           checkpoint_every=10, resume=True, normalize_per_prompt=True)
dt = time.time() - t0
log(f"FIT DONE in {dt:.0f}s = {dt/3600:.2f}h ({dt/max(lens.n_prompts,1):.0f}s/prompt)")
path = f"{OUT}/glm52_band_n{lens.n_prompts}_s{MAX_SEQ}.pt"
lens.save(path)
log(f"saved {path}")
log("DONE")
