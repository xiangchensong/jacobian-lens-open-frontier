"""Readout overlap: compare the three dumps dump_readout.py wrote. No GPU, no model.

For each band layer, the overlap between the top-k sets of two runs at the same
(prompt, position). Reported against the floor: same model, same prompts, a lens
fitted on disjoint text. Divergence only counts as a training effect if it
exceeds what changing the fitting corpus already produces.
"""
import json
import os
import time

import torch

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
R = {k: torch.load(f"{D}/readout_{k}.pt") for k in ("preview", "cur", "cur_disj")}
BAND = sorted(set.intersection(*[set(r) for r in R.values()]))

def jac(a, b, k):
    """Mean top-k Jaccard overlap over positions."""
    n = min(a.shape[0], b.shape[0])
    A, B = a[:n, :k], b[:n, :k]
    out = []
    for i in range(n):
        s1, s2 = set(A[i].tolist()), set(B[i].tolist())
        out.append(len(s1 & s2) / len(s1 | s2))
    return sum(out) / len(out)

rows = []
for K in (1, 5, 10):
    log(f"=== top-{K} overlap ===")
    for l in BAND:
        p = jac(R["preview"][l], R["cur"][l], K)
        f = jac(R["cur_disj"][l], R["cur"][l], K)
        rows.append(dict(k=K, layer=l, pair=p, floor=f, excess=f - p))
        log(f"  L{l:<3} pair={p:.4f}  floor={f:.4f}  excess={f-p:+.4f}")
    sel = [r for r in rows if r["k"] == K]
    lo = [r for r in sel if r["layer"] < BAND[len(BAND)//2]]
    hi = [r for r in sel if r["layer"] >= BAND[len(BAND)//2]]
    mlo = sum(r["excess"] for r in lo)/len(lo)
    mhi = sum(r["excess"] for r in hi)/len(hi)
    log(f"  top-{K}: mean excess {sum(r['excess'] for r in sel)/len(sel):+.4f} "
        f"| early {mlo:+.4f} late {mhi:+.4f} -> "
        f"{'late-concentrated' if mhi > mlo else 'early-concentrated'}")
json.dump(rows, open(f"{D}/b2_compare.json", "w"), indent=1)
log("DONE")
