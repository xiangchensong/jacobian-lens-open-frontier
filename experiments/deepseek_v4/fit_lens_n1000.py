"""Fit the band lens at the paper's corpus size (n=1000, L19-39).

EXPECTED RESULT: no change. The n-scaling curve measured on 3.3 is flat from
n=60 to n=100 (0.434 -> 0.434 base-conditional), so the estimator has already
converged and 1000 prompts should land in the same place. This is run anyway
because it definitively closes the "your lens is undertrained" objection to the
swap shortfalls, which is worth ~26 GPU-hours in a replication. If it does move
the numbers, that is a genuine surprise and the n-scaling measurement was
misleading -- either way the outcome is informative and will be reported as
measured rather than as hoped.

Band-restricted to L19-39 (set by Experiment 6.2) because the retained graph
spans from min(source_layers) onward: fitting L0-42 would cost roughly twice as
much for layers no experiment reads. Checkpoints every 25 prompts -- each is
len(source_layers) * d_target * d_source * 4 bytes = 21 * 4096 * 16384 * 4
= 5.6 GB, so checkpointing every prompt would dominate the run.
"""
# NOTE: paths under /data3 (model weights, fitted lenses in /data3/fan-test/jlens_out)
# are machine-specific to the original 8xB200 host; adjust for your environment.
import time

import torch

MODEL = "/data3/fan-test/models/DeepSeek-V4-Flash-0731"
OUT = "/data3/fan-test/jlens_out"
CKPT = f"{OUT}/band_n1000_ckpt.pt"
SOURCE = list(range(19, 40))
DIM_BATCH = 16
MAX_SEQ = 128
N_PROMPTS = 1000


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


from transformers import AutoModelForCausalLM, AutoTokenizer, FineGrainedFP8Config

n_gpu = torch.cuda.device_count()
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL, dtype="auto", device_map="auto",
    max_memory={i: "146GiB" for i in range(n_gpu)},
    attn_implementation="eager", experts_implementation="grouped_mm",
    quantization_config=FineGrainedFP8Config(dequantize=True))
model.eval()
tok = AutoTokenizer.from_pretrained(MODEL)
log(f"LOADED in {time.time()-t0:.0f}s on {n_gpu} GPUs")

import jlens.examples as ex
from jlens._logging import configure_logging
from jlens.fitting import fit
from jlens.hf import from_hf

# Without this the library's per-prompt progress is logged at INFO and swallowed
# by Python's default WARNING level -- which on a 26-hour run means no way to
# tell slow progress from a stall, and nothing for the monitor to match on.
configure_logging()

lm = from_hf(model, tok)
# oversample: load_wikitext_prompts filters by length, so ask for more than
# needed and truncate, rather than discovering a short corpus mid-run
prompts = ex.load_wikitext_prompts(N_PROMPTS, min_chars=600)
log(f"corpus: {len(prompts)} prompts (asked {N_PROMPTS})")
if len(prompts) < N_PROMPTS:
    log(f"WARNING: only {len(prompts)} available; fitting on those")

t0 = time.time()
lens = fit(
    lm, prompts,
    source_layers=SOURCE,
    dim_batch=DIM_BATCH,
    max_seq_len=MAX_SEQ,
    checkpoint_path=CKPT,
    checkpoint_every=25,
    resume=True,
    normalize_per_prompt=True,
)
dt = time.time() - t0
log(f"FIT DONE in {dt:.0f}s = {dt/3600:.2f}h "
    f"({dt/max(lens.n_prompts,1):.0f}s/prompt)")
path = f"{OUT}/v4flash_band_n{lens.n_prompts}_s{MAX_SEQ}.pt"
lens.save(path)
log(f"saved {path}")
log("DONE")
