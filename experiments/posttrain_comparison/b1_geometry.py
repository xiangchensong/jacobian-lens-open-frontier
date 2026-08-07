"""Lens geometry: where in depth the two post-trainings differ.

Per band layer, compares the preview's J_l against 0731's three ways:
  CKA           on lens vectors for a fixed vocabulary sample (representation
                similarity, invariant to rotation/scale)
  principal     between the top-k left singular subspaces (how much of the
  angles        readout subspace is shared)
  cosine        of the flattened matrices (crude but interpretable)

The number that matters is not the raw similarity -- two lenses of the SAME
checkpoint fitted on disjoint corpus halves are also not identical. So every
statistic is reported alongside the same statistic computed between 0731's main
lens and 0731's disjoint-corpus lens. Only divergence exceeding that floor is
attributable to post-training rather than to which prompts we happened to fit on.
"""
# NOTE: paths under /data3 (model weights, fitted lenses in /data3/fan-test/jlens_out)
# are machine-specific to the original 8xB200 host; adjust for your environment.
import json
import os
import time

import torch

from jlens.lens import JacobianLens

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
K = 64          # subspace rank for principal angles
NVOCAB = 2048   # vocabulary sample for CKA
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

P = {"preview":  "/data3/fan-test/jlens_out/v4preview_band_n100_s128.pt",
     "cur":      "/data3/fan-test/jlens_out/v4flash_runAnorm_n100_s128.pt",
     "cur_disj": "/data3/fan-test/jlens_out/v4flash_disjoint100_s128.pt"}
L = {k: JacobianLens.load(v) for k, v in P.items()}
for k, l in L.items():
    ls = sorted(l.jacobians)
    log(f"  {k}: n={l.n_prompts} L{ls[0]}-{ls[-1]} ({len(ls)})")
BAND = sorted(set(L["preview"].jacobians) & set(L["cur"].jacobians) & set(L["cur_disj"].jacobians))
log(f"  common band: L{BAND[0]}-{BAND[-1]} ({len(BAND)} layers)")

g = torch.Generator().manual_seed(0)
# fixed unembedding-row sample, shared across every layer and pair
W = None
try:
    import safetensors.torch as st  # noqa
except Exception:
    pass

def cka(X, Y):
    """Linear CKA between two [n, d] representations."""
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    xy = (X.T @ Y).norm() ** 2
    xx = (X.T @ X).norm()
    yy = (Y.T @ Y).norm()
    return float(xy / (xx * yy + 1e-12))

def principal(A, B, k=K):
    """Mean cosine of principal angles between the top-k left subspaces."""
    Ua = torch.linalg.svd(A, full_matrices=False).U[:, :k]
    Ub = torch.linalg.svd(B, full_matrices=False).U[:, :k]
    s = torch.linalg.svdvals(Ua.T @ Ub).clamp(0, 1)
    return float(s.mean())

# a fixed random probe basis stands in for the vocabulary sample: J acts on the
# 16384-d source, and CKA on J @ probes measures how the two maps treat the same
# inputs without needing the model's unembedding matrix here.
probes = torch.randn(NVOCAB, 16384, generator=g).to(DEV)
probes = probes / probes.norm(dim=-1, keepdim=True)

rows = []
for l in BAND:
    J = {k: L[k].jacobians[l].to(DEV, torch.float32) for k in L}
    R = {k: probes @ J[k].T for k in J}           # [NVOCAB, 4096]
    r = dict(layer=l,
             cka_pair=cka(R["preview"], R["cur"]),
             cka_floor=cka(R["cur"], R["cur_disj"]),
             pa_pair=principal(J["preview"], J["cur"]),
             pa_floor=principal(J["cur"], J["cur_disj"]),
             cos_pair=float(torch.nn.functional.cosine_similarity(
                 J["preview"].flatten(), J["cur"].flatten(), dim=0)),
             cos_floor=float(torch.nn.functional.cosine_similarity(
                 J["cur"].flatten(), J["cur_disj"].flatten(), dim=0)))
    r["cka_excess"] = r["cka_floor"] - r["cka_pair"]   # >0 means real divergence
    rows.append(r)
    log(f"  L{l:<3} CKA pair={r['cka_pair']:.4f} floor={r['cka_floor']:.4f} "
        f"excess={r['cka_excess']:+.4f} | PA pair={r['pa_pair']:.4f} floor={r['pa_floor']:.4f}")
    del J, R
    torch.cuda.empty_cache()

json.dump(rows, open(f"{OUT}/b1_geometry.json", "w"), indent=1)
ex = [r["cka_excess"] for r in rows]
log(f"  mean CKA excess over floor: {sum(ex)/len(ex):+.4f}")
lo, hi = BAND[:len(BAND)//2], BAND[len(BAND)//2:]
mlo = sum(r["cka_excess"] for r in rows if r["layer"] in lo)/len(lo)
mhi = sum(r["cka_excess"] for r in rows if r["layer"] in hi)/len(hi)
log(f"  early band L{lo[0]}-{lo[-1]}: {mlo:+.4f}   late band L{hi[0]}-{hi[-1]}: {mhi:+.4f}")
log(f"  PREDICTION 2 (divergence concentrates late): "
    f"{'SUPPORTED' if mhi > mlo else 'NOT SUPPORTED'}")
log("DONE")
