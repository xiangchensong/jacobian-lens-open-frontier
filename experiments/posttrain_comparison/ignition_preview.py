"""Experiments 6.3 (ignition: how sharply the band commits to one reading of an
ambiguous token) and 1.2 (verbal introspection) on the preview checkpoint.
Path-only derivation of deepseek_v4/ignition.py.

1.2 verbal introspection
    The model is told a thought may have been injected and asked to identify it.
    One of the prefills is teacher-forced, ending in an open quote, so the next
    predicted token is the reported word. The concept's lens direction is
    steered into every band layer at every token of the user turn; strength 0 is
    the control. Score = rank of the surface form at the open quote. The paper
    reports median reciprocal rank against strength.

6.3 ignition
    A different kind of intervention entirely -- no lens involved. The carrier's
    {W} token *embedding* is replaced by alpha*emb(A) + (1-alpha)*emb(B) and
    alpha is swept 0 -> 1. At each layer we read A's reciprocal-rank share,
       1/rank(A) / (1/rank(A) + 1/rank(B)),
    at the {W} position. Pre-workspace layers should track alpha smoothly;
    from workspace onset the paper reports a sharp threshold -- the commitment
    to one interpretation that motivates the "ignition" name. The measurable is
    the per-layer 10->90% transition width in alpha: wide early, narrow in band.
"""
# NOTE: paths under /data3 (model weights, fitted lenses in /data3/fan-test/jlens_out)
# are machine-specific to the original 8xB200 host; adjust for your environment.

import json
import os
import sys
import time

import torch

MODEL = "/data3/fan-test/models/DeepSeek-V4-Flash"
LENS = "/data3/fan-test/jlens_out/v4preview_band_n100_s128.pt"
BAND = list(range(19, 40))
DEV = "cuda:0"
SCRATCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(SCRATCH, exist_ok=True)
STRENGTHS = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)
N_CONCEPTS = 40
ALPHAS = [i / 10 for i in range(11)]
N_PAIRS, N_CARRIERS = 8, 4


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
log(f"LOADED in {time.time()-t0:.0f}s")

sys.path.insert(0, f"{MODEL}/encoding")
import encoding_dsv4 as DSV4

from jlens.hf import from_hf
from jlens.hooks import ActivationRecorder
from jlens.intervene import steer
from jlens.lens import JacobianLens

lm = from_hf(model, tok)
lens = JacobianLens.load(LENS)
lens.jacobians = {l: lens.jacobians[l].to(DEV) for l in BAND}
lens.source_layers = BAND
ALL_LAYERS = list(range(lm.n_layers))


def single_id(w):
    for v in (" " + w, w, " " + w.capitalize(), w.capitalize()):
        e = tok.encode(v, add_special_tokens=False)
        if len(e) == 1:
            return e[0]
    return None


# ==================================================== 1.2 verbal introspection
def exp12():
    log("=" * 70)
    log("(1.2) verbal introspection -- reported word vs injection strength")
    d = json.load(open("data/experiments/verbal-introspection.json"))
    intro = d["intro_prompt"]
    prefill = d["prefills"]["word"]
    concepts = [c for c in d["concepts"] if single_id(c["surface"]) is not None]
    concepts = concepts[:N_CONCEPTS]
    log(f"  {len(concepts)} single-token concepts of {len(d['concepts'])}")

    msgs = [{"role": m["role"], "content": m["content"]} for m in intro]
    base_text = DSV4.encode_messages(msgs, thinking_mode="chat")
    n_user = len(tok.encode(base_text, add_special_tokens=False))
    prompt = base_text + prefill

    res = {s: [] for s in STRENGTHS}
    t0 = time.time()
    for ci, c in enumerate(concepts):
        tid = single_id(c["surface"])
        ids = lm.encode(prompt, max_length=512)
        user_pos = list(range(min(n_user, ids.shape[1])))
        for s in STRENGTHS:
            ctx = steer(lens, lm, layers=BAND, token=tid, strength=s,
                        positions=user_pos)
            with torch.no_grad(), ctx:
                lg = model(ids).logits[0, -1].float()
            res[s].append(1.0 / (1 + int((lg > lg[tid]).sum())))
        if (ci + 1) % 10 == 0:
            log(f"    {ci+1}/{len(concepts)}  {time.time()-t0:.0f}s")
    log(f"  {'strength':>9} {'median RR':>10} {'rank1 rate':>11}")
    for s in STRENGTHS:
        v = sorted(res[s])
        med = v[len(v) // 2]
        top1 = sum(1 for x in res[s] if x == 1.0) / len(v)
        log(f"  {s:>9} {med:>10.4f} {top1:>11.3f}")
    log("  paper: median reciprocal rank rises with strength; control (0) at floor")
    json.dump({str(s): res[s] for s in STRENGTHS},
              open(f"{SCRATCH}/exp12_introspection.json", "w"), indent=1)


# ==================================================== 6.3 ignition
class MixEmbedding:
    """Replace one position's input embedding with a mix of two tokens."""

    def __init__(self, model, pos, id_a, id_b, alpha):
        self.emb = model.get_input_embeddings()
        self.pos, self.a, self.b, self.alpha = pos, id_a, id_b, alpha
        self.h = None

    def __enter__(self):
        w = self.emb.weight

        def hook(mod, inp, out):
            v = (self.alpha * w[self.a] + (1 - self.alpha) * w[self.b])
            out = out.clone()
            out[:, self.pos] = v.to(out.dtype)
            return out

        self.h = self.emb.register_forward_hook(hook)
        return self

    def __exit__(self, *e):
        if self.h:
            self.h.remove()


@torch.no_grad()
def exp63():
    log("=" * 70)
    log("(6.3) ignition -- commitment to one reading as the mixture is swept")
    d = json.load(open("data/experiments/ignition.json"))
    countries = [c for c in d["countries_12"] if single_id(c) is not None]
    pairs = [(countries[i], countries[j])
             for i in range(len(countries)) for j in range(i + 1, len(countries))]
    pairs = pairs[:N_PAIRS]
    carriers = d["ctx_templates"][:N_CARRIERS]
    log(f"  {len(pairs)} pairs x {len(carriers)} carriers x {len(ALPHAS)} alphas")

    # share[layer][alpha] accumulated over trials
    acc = {l: {a: [] for a in ALPHAS} for l in ALL_LAYERS}
    t0 = time.time()
    n = 0
    for (A, B) in pairs:
        ia, ib = single_id(A), single_id(B)
        for carrier in carriers:
            text = carrier.replace("{W}", A)
            ids = lm.encode(text, max_length=64)
            toks = ids[0].tolist()
            try:
                pos = len(toks) - 1 - toks[::-1].index(ia)
            except ValueError:
                continue
            n += 1
            for a in ALPHAS:
                with MixEmbedding(model, pos, ia, ib, a):
                    with ActivationRecorder(lm.layers, at=ALL_LAYERS) as rec:
                        lm.forward(ids)
                        acts = {l: rec.activations[l][0, pos].detach()
                                for l in ALL_LAYERS}
                for l in ALL_LAYERS:
                    H = acts[l].to(DEV).float()
                    lg = lm.unembed(H.unsqueeze(0).unsqueeze(0)).float().reshape(-1)
                    ra = 1 + int((lg > lg[ia]).sum())
                    rb = 1 + int((lg > lg[ib]).sum())
                    acc[l][a].append((1 / ra) / ((1 / ra) + (1 / rb)))
        log(f"    pair {A}/{B} done, {n} trials, {time.time()-t0:.0f}s")

    log(f"  {'layer':>5} " + "".join(f"a={a:<4.1f}" for a in ALPHAS) + "  width10-90")
    out = {}
    for l in ALL_LAYERS:
        means = [sum(acc[l][a]) / len(acc[l][a]) if acc[l][a] else float("nan")
                 for a in ALPHAS]
        lo = hi = None
        for a, m in zip(ALPHAS, means, strict=False):
            if lo is None and m >= 0.1:
                lo = a
            if m >= 0.9 and hi is None:
                hi = a
        width = (hi - lo) if (lo is not None and hi is not None) else float("nan")
        out[l] = {"means": means, "width": width}
        if l % 3 == 0 or l in (19, 39):
            log(f"  {l:>5} " + "".join(f"{m:>6.2f}" for m in means)
                + f"  {width:>10.2f}")
    log("  paper: smooth tracking pre-workspace, sharp threshold from onset")
    json.dump({str(k): v for k, v in out.items()},
              open(f"{SCRATCH}/exp63_ignition_preview.json", "w"), indent=1)


for fn in (exp12, exp63):
    try:
        fn()
    except Exception as exc:
        import traceback
        log(f"!! {fn.__name__} FAILED {type(exc).__name__}: {exc}")
        traceback.print_exc()

log("DONE")
