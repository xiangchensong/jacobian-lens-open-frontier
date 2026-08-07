"""Experiment 3.3 -- the multi-hop swap: replace an unspoken intermediate's lens
coordinate and test whether the model's answer follows.

90 two-hop prompts. Each names an unspoken bridge entity (`intermediate`) that
the model must infer to answer. We swap that entity's lens coordinate for
`swap_to`'s across the workspace band at every prompt position, and ask whether
the model's answer follows to `swap_answer`.

Published targets: Haiku 4.5 54%, Sonnet 4.5 70%, Opus 4.5 70%.

If V4 lands far outside the paper's range the
question is whether it is a real architecture difference or a bug in
jlens.intervene, and nothing else should run until that is settled -- so this
script also reports the diagnostics needed to tell those apart:

  * baseline greedy accuracy (is the model even solving the task?)
  * swap coverage (how many items have single-token directions at all?)
  * the intermediate's own lens rank before the swap (is the concept there to
    be swapped?)
  * alpha=1 and alpha=2, the paper's standard and double-strength conditions
"""
# NOTE: paths under /data3 (model weights, fitted lenses in /data3/fan-test/jlens_out)
# are machine-specific to the original 8xB200 host; adjust for your environment.

import json
import time

import torch

MODEL = "/data3/fan-test/models/DeepSeek-V4-Flash-0731"
LENS = "/data3/fan-test/jlens_out/v4flash_runAnorm_n100_s128.pt"
BAND = list(range(19, 40))          # set by Experiment 6.2
DEV = "cuda:0"
ALPHAS = (1.0, 2.0)


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

from jlens.hf import from_hf
from jlens.hooks import ActivationRecorder
from jlens.intervene import swap
from jlens.lens import JacobianLens

lm = from_hf(model, tok)
lens = JacobianLens.load(LENS)
# keep only the band; lens_directions is indexed per layer so this is free
lens.jacobians = {l: lens.jacobians[l].to(DEV) for l in BAND}
lens.source_layers = BAND
log(f"lens restricted to band L{BAND[0]}-{BAND[-1]}")

items = json.load(open("data/experiments/probe-swap.json"))["items"]
log(f"{len(items)} items")


def first_id(word):
    """Leading token of ` word`, the form these appear as mid-sentence."""
    for v in (" " + word, word, " " + word.capitalize(), word.capitalize()):
        enc = tok.encode(v, add_special_tokens=False)
        if enc:
            return enc[0]
    return None


def single_id(word):
    """Single-token id if the word is one token, else None."""
    for v in (" " + word, word, " " + word.capitalize(), word.capitalize()):
        enc = tok.encode(v, add_special_tokens=False)
        if len(enc) == 1:
            return enc[0]
    return None


@torch.no_grad()
def greedy_next(ids, ctx=None):
    if ctx is None:
        out = model(ids)
    else:
        with ctx:
            out = model(ids)
    return int(out.logits[0, -1].argmax())


@torch.no_grad()
def intermediate_rank(ids, tid):
    """Best band rank of the intermediate before any intervention."""
    with ActivationRecorder(lm.layers, at=BAND) as rec:
        lm.forward(ids)
        acts = {l: rec.activations[l][0].detach() for l in BAND}
    best = None
    for l in BAND:
        H = acts[l].to(DEV).float()
        lg = lm.unembed(H.flatten(1) @ lens.jacobians[l].T).float()
        r = int((lg > lg[:, tid:tid + 1]).sum(dim=-1).min())
        best = r if best is None else min(best, r)
    return best


stats = {a: {"hit": 0, "n": 0} for a in ALPHAS}
base_ok = 0
scored = 0
skipped = 0
ranks = []

t0 = time.time()
for i, it in enumerate(items):
    src = single_id(it["intermediate"])
    dst = single_id(it["swap_to"])
    if src is None or dst is None:
        skipped += 1
        continue
    ans_id = first_id(it["answer"])
    swp_id = first_id(it["swap_answer"])
    if ans_id is None or swp_id is None:
        skipped += 1
        continue

    ids = lm.encode(it["prompt"], max_length=256)
    scored += 1
    if greedy_next(ids) == ans_id:
        base_ok += 1
    try:
        ranks.append(intermediate_rank(ids, src))
    except Exception as exc:
        log(f"  rank failed on {it['name']}: {type(exc).__name__}: {exc}")

    for a in ALPHAS:
        ctx = swap(lens, lm, layers=BAND, from_token=src, to_token=dst, alpha=a)
        try:
            got = greedy_next(ids, ctx)
        except Exception as exc:
            log(f"  swap failed on {it['name']} a={a}: {type(exc).__name__}: {exc}")
            continue
        stats[a]["n"] += 1
        if got == swp_id:
            stats[a]["hit"] += 1

    if (i + 1) % 15 == 0:
        el = time.time() - t0
        log(f"  {i+1}/{len(items)}  {el:.0f}s  "
            + "  ".join(f"a={a}: {stats[a]['hit']}/{stats[a]['n']}" for a in ALPHAS))

log("=" * 70)
log(f"EXPERIMENT 3.3 -- multi-hop swap, band L{BAND[0]}-{BAND[-1]}")
log(f"  items scored           {scored}/{len(items)}  (skipped {skipped}: "
    f"intermediate or swap_to not single-token)")
log(f"  baseline greedy == answer   {base_ok}/{scored} = {base_ok/max(scored,1):.3f}")
if ranks:
    rs = sorted(ranks)
    log(f"  intermediate band rank      median {rs[len(rs)//2]}  "
        f"rank<=10 on {sum(1 for r in rs if r <= 10)}/{len(rs)}")
for a in ALPHAS:
    s = stats[a]
    log(f"  alpha={a}: swap success   {s['hit']}/{s['n']} = "
        f"{s['hit']/max(s['n'],1):.3f}    (paper: 0.54-0.70)")
log("DONE")
