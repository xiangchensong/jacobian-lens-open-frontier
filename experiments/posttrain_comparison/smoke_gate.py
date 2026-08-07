"""Verify the preview and 0731 checkpoints are actually comparable, before
anything is measured on either.

Three things must hold, and this script exits non-zero if any fails:

  1. The preview's main stack loads as 43 blocks with d_model 4096 and hc_mult 4,
     i.e. the MTP difference (1 block vs 3) really is dropped at load.
  2. The chat encoding is byte-identical between the two checkpoints for the
     prompts we will use. 0731 RENAMED the reasoning-effort levels -- the
     preview's `max` prefix became 0731's `high` -- so setting the level by NAME
     silently changes the prompt. This gate compares encoded strings directly.
  3. A forward pass runs and the graph is retained, so the fit will work.

Failing loudly here is the point: better no run than a comparison of two
different prompts.
"""
# NOTE: paths under /data3 (model weights, fitted lenses in /data3/fan-test/jlens_out)
# are machine-specific to the original 8xB200 host; adjust for your environment.
import sys
import time

import torch

PREV = "/data3/fan-test/models/DeepSeek-V4-Flash"
CUR  = "/data3/fan-test/models/DeepSeek-V4-Flash-0731"
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

# ---- gate 2 first: pure text, no GPU needed -------------------------------
import importlib.util


def load_enc(root):
    spec = importlib.util.spec_from_file_location(
        f"enc_{abs(hash(root))}", f"{root}/encoding/encoding_dsv4.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m
ep, ec = load_enc(PREV), load_enc(CUR)
MSGS = [{"role": "user", "content": "The capital of France is"}]
bad = 0
for mode in ("chat", "thinking"):
    try:
        a = ep.encode_messages(MSGS, thinking_mode=mode)
        b = ec.encode_messages(MSGS, thinking_mode=mode)
    except TypeError:
        a, b = ep.encode_messages(MSGS), ec.encode_messages(MSGS)
    same = a == b
    log(f"  thinking_mode={mode!r}: encodings {'IDENTICAL' if same else 'DIFFER'}")
    if not same:
        bad += 1
        for i,(x,y) in enumerate(zip(a, b, strict=False)):
            if x != y:
                log(f"    first divergence at char {i}: {a[max(0,i-40):i+40]!r}")
                log(f"                            vs  {b[max(0,i-40):i+40]!r}")
                break
        log(f"    len {len(a)} vs {len(b)}")
if bad:
    log("GATE 2 FAIL: prompts would differ between checkpoints. STOP.")
    sys.exit(1)
log("GATE 2 PASS: chat encoding byte-identical for the modes we use")

# ---- gates 1 and 3 --------------------------------------------------------
from transformers import AutoModelForCausalLM, AutoTokenizer, FineGrainedFP8Config

n_gpu = torch.cuda.device_count()
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    PREV, dtype="auto", device_map="auto",
    max_memory={i: "146GiB" for i in range(n_gpu)},
    attn_implementation="eager", experts_implementation="grouped_mm",
    quantization_config=FineGrainedFP8Config(dequantize=True))
model.eval()
tok = AutoTokenizer.from_pretrained(PREV)
log(f"loaded preview in {time.time()-t0:.0f}s")
from jlens.hf import from_hf
from jlens.hooks import ActivationRecorder
from jlens.protocol import d_source_of, d_target_of, target_module_of

lm = from_hf(model, tok)
assert lm.n_layers == 43, f"expected 43 blocks, got {lm.n_layers}"
assert lm.d_model == 4096, lm.d_model
assert d_source_of(lm) == 16384, d_source_of(lm)
log(f"GATE 1 PASS: n_layers={lm.n_layers} d_model={lm.d_model} "
    f"d_source={d_source_of(lm)} d_target={d_target_of(lm)}")

ids = lm.encode("The capital of France is", max_length=32)
TARGET = lm.n_layers
with ActivationRecorder(lm.layers, at=[19, TARGET], start_graph_at=19,
                        extra_modules={TARGET: target_module_of(lm)}) as rec:
    lm.forward(ids.expand(2, -1))
    tgt = rec.activations[TARGET]
    g = torch.autograd.grad(tgt, rec.activations[19],
                            grad_outputs=torch.zeros_like(tgt).index_fill_(
                                2, torch.tensor([0], device=tgt.device), 1.0))[0]
ok = torch.isfinite(g).all() and g.abs().sum() > 0
log(f"GATE 3 {'PASS' if ok else 'FAIL'}: backward through preview, "
    f"grad norm {float(g.norm()):.3e}")
sys.exit(0 if ok else 1)
