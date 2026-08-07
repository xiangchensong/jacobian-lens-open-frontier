"""Fit a Jacobian lens on DeepSeek-V4-Flash-0731: all 43 layers, S=128, 100 WikiText prompts.

Probes memory on prompt 1 before committing ~8 h: the retained graph spans from
min(source_layers), so going 27 -> 43 layers grows the graph share on the fullest
card and the dim_batch=16 that Phase 2 used is expected to OOM. Falls back a step
rather than dying hours in.

Resumable: checkpoint_path + resume=True, so a crash costs at most 10 prompts.
"""
# NOTE: paths under /data3 (model weights, fitted lenses in /data3/fan-test/jlens_out)
# are machine-specific to the original 8xB200 host; adjust for your environment.

import logging
import os
import time

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MODEL = "/data3/fan-test/models/DeepSeek-V4-Flash-0731"
OUT = "/data3/fan-test/jlens_out"
SOURCE_LAYERS = list(range(0, 43))
MAX_SEQ = 128
N_PROMPTS = 100
CKPT = f"{OUT}/runAnorm_ckpt.pt"

os.makedirs(OUT, exist_ok=True)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


from transformers import AutoModelForCausalLM, AutoTokenizer, FineGrainedFP8Config

n_gpu = torch.cuda.device_count()
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype="auto", device_map="auto",
    max_memory={i: "146GiB" for i in range(n_gpu)},
    attn_implementation="eager", experts_implementation="grouped_mm",
    quantization_config=FineGrainedFP8Config(dequantize=True),
)
model.eval()
tok = AutoTokenizer.from_pretrained(MODEL)
log(f"LOADED in {time.time()-t0:.0f}s")

from jlens.examples import load_wikitext_prompts
from jlens.fitting import fit, jacobian_for_prompt
from jlens.hf import from_hf

lens_model = from_hf(model, tok)
prompts = load_wikitext_prompts(N_PROMPTS)
log(f"corpus: {len(prompts)} prompts")

# ---- pick the largest dim_batch that fits all 43 layers ----
DIM_BATCH = None
for candidate in (16, 8, 4):
    for i in range(n_gpu):
        torch.cuda.reset_peak_memory_stats(i)
    try:
        t0 = time.time()
        J, _, _ = jacobian_for_prompt(
            lens_model, prompts[0], SOURCE_LAYERS,
            dim_batch=candidate, max_seq_len=MAX_SEQ,
        )
        dt = time.time() - t0
        peak = {i: round(torch.cuda.max_memory_allocated(i) / 1e9, 1) for i in range(n_gpu)}
        log(f"PROBE dim_batch={candidate}: {dt:.0f}s/prompt  peak={peak}")
        del J
        DIM_BATCH = candidate
        break
    except torch.cuda.OutOfMemoryError:
        log(f"PROBE dim_batch={candidate}: OOM")
        for _ in range(n_gpu):
            torch.cuda.empty_cache()

if DIM_BATCH is None:
    log("PROBE FAILED: even dim_batch=4 OOMs across 43 layers")
    raise SystemExit(1)

log(f"CHOSE dim_batch={DIM_BATCH}; starting {N_PROMPTS}-prompt fit")

# 43 layers x 4096 x 16384 x fp32 = 11.5 GB per checkpoint -> every 10.
t0 = time.time()
lens = fit(
    lens_model, prompts,
    source_layers=SOURCE_LAYERS,
    dim_batch=DIM_BATCH,
    max_seq_len=MAX_SEQ,
    checkpoint_path=CKPT,
    checkpoint_every=10,
    resume=True,
    normalize_per_prompt=True,
)
dt = time.time() - t0
log(f"FIT DONE in {dt:.0f}s ({dt/max(lens.n_prompts,1):.0f}s/prompt)")

path = f"{OUT}/v4flash_runAnorm_n{lens.n_prompts}_s{MAX_SEQ}.pt"
lens.save(path)
log(f"SAVED {path} ({os.path.getsize(path)/1e9:.1f} GB)  {lens!r}")
log("DONE")
