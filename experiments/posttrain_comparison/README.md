# Post-training comparison: DeepSeek-V4-Flash preview vs 0731

DeepSeek shipped two post-trainings of one reported base — the preview (04-22) and 0731,
described by DeepSeek as *"keeps the same model architecture and size … and was only
re-post-trained"* — which makes the pair a vendor-described controlled experiment on the
training recipe (identical architecture is verified from the shards; identical pretrained
weights are taken from DeepSeek's description), orthogonal to the cross-family axis the
main write-up runs on. Every statistic here
is floor-controlled: reported against the same statistic computed between 0731's main lens
and a second 0731 lens fitted on disjoint corpus prompts, so a difference only counts as a
training effect if it exceeds what changing the fitting corpus alone already produces.

Headline: the redo **changed the token-level lens readout far more than the lens
geometry** — cross-checkpoint top-1 readout overlap is 0.293 against a same-checkpoint
disjoint-corpus floor of 0.592, while mean CKA is 0.821 against a floor of 0.921 — and
every structural property (band, ignition, selectivity, white-bear suppression, J-space
compression) is unchanged.

| script | what it measures | key number it backs |
|---|---|---|
| `smoke_gate.py` | go/no-go before any measurement: preview loads as 43 blocks, chat encodings byte-identical between checkpoints (0731 renamed the reasoning-effort levels — preview `max` ≡ 0731 `high` — so this had to be asserted, not assumed), backward graph retained | every prompt in this directory is comparable across checkpoints |
| `fit_lens_preview.py` | fits the preview band lens, recipe-matched to 0731's n=100 fit (same corpus slice, band L19–39, S=128, per-prompt normalization) | the lens every preview row below uses (2.53 h, converged to max_d_mean 1.44e-02) |
| `b1_geometry.py` | per-band-layer CKA, principal angles and cosine between the two lenses, each beside its disjoint-corpus floor | mean CKA **0.821 vs floor 0.921**; CKA excess **+0.1002**, concentrated early/mid-band (+0.1192 L19–28 vs +0.0829 L29–39) |
| `dump_readout.py` | dumps each checkpoint's own top-50 readout on 20 fixed WikiText prompts; run once per variant (`preview` / `cur` / `cur_disj`) | input to `b2_readout_overlap.py` (writes `results/readout_*.pt`, gitignored) |
| `b2_readout_overlap.py` | top-k readout overlap at identical (prompt, position), pair vs floor; offline, no GPU | top-1 overlap **0.293 vs floor 0.592** (top-5 0.25 vs 0.49, top-10 0.26 vs 0.47), with **no depth profile** |
| `phaseA_preview.py` | 5.3 position-adaptive ablation + matched-rank random control + selectivity, 3.3 multi-hop swap, 3.5 arithmetic ordering/clearing, on the preview | selectivity **0.529 vs 0.508** — three post-trainings (with GLM's 0.498) in a 0.50–0.53 band against Claude's >0.90 |
| `section8_preview.py` | the four self-monitoring probes read through each checkpoint's own lens, both sides chat-encoded | the redo did not add or remove any of the four signals (**0/0, 8→9, 0/0, 0/0**); roleplay-fictional was installed by the *first* post-training |
| `workspace_band_preview.py` † | 6.2 onset/offset statistics, preview read with its own band lens | band shape preserved, top-1 converging at L39 (0.1687 vs 0.1665) |
| `workspace_band_floor.py` † | the same statistics, 0731 read with the disjoint-corpus lens | the floor that keeps top-1 as a result (2.8× floor) and downgrades effective dimensionality to suggestive (1.5×, direction shared by the floor) |
| `ignition_preview.py` † | 6.3 ignition sweep + 1.2 introspection on the preview | 10→90% crossings span **exactly L19–39 in both checkpoints, layer for layer** — the band did not move |
| `evaluate_lens_preview.py` † | six lens-quality sets on the preview (invoke with the preview band lens) | **three sets up, three down** — a mixed profile, i.e. different contents rather than a worse lens, so the readout result stands |
| `modulation_preview.py` † | 2.2 white bear at generated positions, verbalization-controlled | suppress **0.000 on both** post-trainings (preview focus 0.833, so the zero is not a floor artifact) |
| `jspace_variance_preview.py` † | 1.3 J-space share of concept-vector variance on the preview | **0.73% vs 0731's 0.4%** — both an order of magnitude under the paper's 6–15%; the compression is recipe-invariant |
| `ablation_preview.py` † | 5.4 experiential and 7.1 honeypot continuations under band ablation with matched-rank random control | both checkpoints lose the experiential register under true J-space ablation and keep it under the random control (failure mode differs; ungraded) |
| `broadcast_preview.py` ‡ re-derived from the fixed `deepseek_v4/broadcast.py`. Step-0 numbers (recorded): 0731 verify 0.146 / **0.208** (bit-for-bit vs the published value, baseline 0.766); preview 0.245 / **0.234** (baseline 0.734, 1.1 = 0.825) — recipe-invariant at the α=2 operating point (0.026 apart, ~0.6 two-sample SE).
| `broadcast_matched_control.py` ‡ | the identical run on 0731 with the n=100 lens | the matched control that exposed the silent clamp fallback (0/192 on both with healthy baselines) |

† **Path-only derivation** of the corresponding `../deepseek_v4/` script: byte-identical
except the model/lens/output-path constants and a first-docstring-line note. The point is
that nothing but the checkpoint under test differs from the published 0731 runs.

‡ **Re-derived from the fixed `../deepseek_v4/broadcast.py`** (grid-selected transfer|2.0
default, hard error on a missing `results/exp33_grid.json`), differing from it only in the
MODEL/LENS constants and the first docstring paragraph. The originals of these two were
derived from the pre-fix script and silently ran mode clamp|1.0 — a formulation the grid
scores 0/53 — so their 0/192 results are struck; the step-0 re-runs' numbers will be
recorded in `results/` when they land.

## Files in `results/`

| file | written by | backs |
|---|---|---|
| `b1_geometry.json` | `b1_geometry.py` | the 89% / +0.1002 geometry numbers (and Fig. 7's per-layer curves) |
| `b2_compare.json` | `b2_readout_overlap.py` | the 49/51/55% readout-retention numbers |
| `phaseA_preview.json` | `phaseA_preview.py` | selectivity 0.529, swap 0.481, clearing on the preview |
| `exp62_stats_preview.json` | `workspace_band_preview.py` | preview band statistics |
| `exp62_stats_floor.json` | `workspace_band_floor.py` | the 6.2 floor (top-1 2.8×, eff-dim 1.5×) |
| `exp63_ignition_preview.json` | `ignition_preview.py` | the identical L19–39 crossing set |

Everything else the scripts write lands in `results/` too (readout dumps as gitignored
`.pt`, section-8/white-bear/1.3 JSONs), but only the six files above are committed.

## Running these

Same environment and caveats as [`../README.md`](../README.md): run from the repo root
(the broadcast pair reads the shared `../results/exp33_grid.json`; several scripts read
`data/experiments/` and `data/evaluations/`), model weights and fitted lenses live under
machine-specific `/data3` paths set at the top of each file. Order matters only where
files flow: `smoke_gate.py` before anything, `fit_lens_preview.py` before any preview
script, `dump_readout.py` three times (`preview`, `cur`, `cur_disj`) before
`b2_readout_overlap.py`.
