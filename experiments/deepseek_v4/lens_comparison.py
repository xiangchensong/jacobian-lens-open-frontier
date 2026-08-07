"""Tuned-lens baseline -- the third method the paper compares against.

The paper's method comparison is J-lens vs logit lens vs **tuned lens**. Only
the logit lens has been run here (the "vanilla" column throughout), so every
J-lens-vs-baseline number so far is missing the strongest baseline.

On this architecture the tuned lens is a particularly clean comparison: it
learns an affine map from the flattened mHC residual (16384) into the collapsed
final basis (4096) — **exactly the shape of J_l**. So J-lens and tuned lens
differ only in how that map is obtained: J_l is a closed-form average of
Jacobians, A_l is fitted by gradient descent on the same corpus. Anything the
tuned lens wins is attributable to learning rather than to a different readout.

Trained the standard way (Belrose et al.): predict the final residual, and
optimise the KL between the resulting logits and the model's own, so the
objective is matched to what the readout is used for.

Saved in JacobianLens format so the existing eval harness can score it with no
changes — it is the same [d_target, d_source] object.
"""
# NOTE: paths under /data3 (model weights, fitted lenses in /data3/fan-test/jlens_out)
# are machine-specific to the original 8xB200 host; adjust for your environment.

import sys
import time

import torch
from torch import nn

MODEL = "/data3/fan-test/models/DeepSeek-V4-Flash-0731"
OUT = "/data3/fan-test/jlens_out/v4flash_tuned_band.pt"
# 3 epochs left KL still falling (2.19 -> 1.46); an undertrained baseline is
# not a baseline, so the reported run uses far more steps.
BAND = list(range(19, 40))
DEV = "cuda:0"
N_PROMPTS = 200
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
LR = 1e-4
SKIP = 16


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


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
from jlens.hf import from_hf
from jlens.hooks import ActivationRecorder
from jlens.lens import JacobianLens

lm = from_hf(model, tok)
D_SRC = getattr(lm, "d_source", lm.d_model)
D_TGT = lm.d_model
TARGET = lm.n_layers          # virtual index of hc_head, same target J_l uses
log(f"tuned lens: {len(BAND)} affine maps {D_SRC} -> {D_TGT}")

# One LINEAR map per band layer -- bias=False deliberately. J_l is a pure linear
# map, so the tuned lens must be one too for the comparison to be like-for-like.
# Training with a bias and then dropping it on save would hand the baseline a
# handicap, which is the same rigged-comparison mistake this report has already
# had to correct once.
maps = {}
for l in BAND:
    A = nn.Linear(D_SRC, D_TGT, bias=False).to(DEV, torch.float32)
    with torch.no_grad():
        A.weight.normal_(0, (1.0 / D_SRC) ** 0.5)
    maps[l] = A
params = [p for A in maps.values() for p in A.parameters()]
opt = torch.optim.Adam(params, lr=LR)
log(f"{sum(p.numel() for p in params)/1e9:.2f} B trainable parameters")

prompts = ex.load_wikitext_prompts(N_PROMPTS, min_chars=600)
log(f"corpus: {len(prompts)} prompts, {EPOCHS} epochs")

extra = {TARGET: lm.target_module} if hasattr(lm, "target_module") else None
t0 = time.time()
step = 0
for epoch in range(EPOCHS):
    tot, seen = 0.0, 0
    for _pi, p in enumerate(prompts):
        ids = lm.encode(p, max_length=128)
        S = ids.shape[1]
        if S < SKIP + 4:
            continue
        with torch.no_grad():
            with ActivationRecorder(lm.layers, at=BAND + [TARGET],
                                    extra_modules=extra) as rec:
                lm.forward(ids)
                acts = {l: rec.activations[l][0].detach() for l in BAND}
                tgt = rec.activations[TARGET][0].detach() if extra else None
            if tgt is None:
                tgt = acts[BAND[-1]]
            ref = lm.unembed(tgt.to(DEV).float()).float().log_softmax(-1)[SKIP:S - 1]

        loss_sum = 0.0
        for l in BAND:
            h = acts[l].to(DEV).float().flatten(1)[SKIP:S - 1]
            pred = lm.unembed(maps[l](h)).float().log_softmax(-1)
            loss = torch.nn.functional.kl_div(
                pred, ref, log_target=True, reduction="batchmean")
            loss_sum = loss_sum + loss
        opt.zero_grad(set_to_none=True)
        loss_sum.backward()
        opt.step()
        tot += float(loss_sum) / len(BAND)
        seen += 1
        step += 1
        if step % 100 == 0:
            log(f"  epoch {epoch} step {step}  mean KL/layer {tot/max(seen,1):.4f}  "
                f"[{time.time()-t0:.0f}s]")
    log(f"epoch {epoch} done: mean KL/layer {tot/max(seen,1):.4f}")

# Save as a JacobianLens so the existing eval harness scores it unchanged --
# it is the same [d_target, d_source] object, no bias to drop.
jac = {l: maps[l].weight.detach().float().cpu() for l in BAND}
lens = JacobianLens(jacobians=jac, n_prompts=len(prompts),
                    d_model=D_TGT, d_source=D_SRC)
lens.save(OUT)
log(f"saved {OUT}")
log("DONE")
