# Replication experiments

Scripts behind [the write-up](https://xiangchensong.github.io/blog/2026/jacobian-lens-global-workspace/), which reports a replication of
*Verbalizable Representations Form a Global Workspace in Language Models* on two open
models: **DeepSeek-V4-Flash-0731** (four-stream hyper-connections, rectangular
`J ∈ ℝ^{4096×16384}`) and **GLM-5.2** (single-stream, square `J ∈ ℝ^{6144×6144}`).

These are the scripts as run, not a rewritten harness — the numbers in the post came out of
this code. One caveat that phrase has to earn: publishing meant repo-relativizing the path
constants, and that once silently changed an estimand — with its grid file missing,
`broadcast.py` fell back to a swap mode the grid itself had rejected. The missing-input
path is now a hard error (see the `broadcast.py` row). The publication rule is **every
script whose numbers appear in the write-up is here**, including the superseded passes the write-up explicitly retracts, so a reader can
check a claim against the code that produced it. Pure diagnostics and exploratory passes that
produced no reported number are not published.

In the tables below, **paper §** is the section number in *Verbalizable Representations…* —
these are the paper's numbering, not the write-up's. **write-up §** names the section of the
post where the number appears.

## DeepSeek-V4-Flash-0731

| script | what it does | paper § | write-up § |
|---|---|---|---|
| `fit_lens.py` | fits the lens, all 43 layers, S=128, n=100 WikiText prompts | — | Appendix (recipe) |
| `fit_lens_n1000.py` | fits the band lens L19–39 at the paper's corpus size, n=1000 | — | "Preregister the prediction you can afford to get wrong" |
| `fit_lens_base.py` | fits the same band lens on DeepSeek-V4-Flash-**Base**, for the section-8 A/B | — | "Base models already carry much of the assistant" |
| `translations.py` | lookup table of non-English forms per concept; imported by `evaluate_lens_multilingual.py` and `workspace_band.py` (both degrade to English-only targets without it) | — | Appendix (deviations) |
| `evaluate_lens.py` | six lens-quality sets, J-lens vs logit-lens baseline, English-only targets | — | method-comparison table |
| `evaluate_lens_multilingual.py` | the same six-set eval with targets expanded through `translations.py` — the run that produced the **J-lens column** of the method-comparison table | — | "The method hierarchy", Fig. 5 |
| `lens_comparison.py` | tuned-lens baseline, trained to convergence, then re-scored on the six sets | — | "The method hierarchy", Fig. 5 |
| `workspace_band.py` | four onset/offset statistics per layer — this is what defines the band L19–39 | 6.2 | "The band itself", Fig. 1 |
| `readout.py` | CKA geometry, arithmetic rank trajectory, modulation and readout selectivity over the band | 6.1, 3.5, 2.2, 5.2 | "The band itself", "Ordered unspoken intermediates: an arithmetic case study", Fig. 4 |
| `ignition.py` | ignition sharpness across the α mixture, plus verbal introspection | 6.3, 1.2 | "Ignition", Fig. 2 |
| `swap.py` | the multi-hop swap at α=1 and α=2 with its diagnostics (baseline accuracy, swap coverage, the intermediate's pre-swap rank) | 3.3 | "Swap and broadcast are not the same quantity" |
| `swap_grid.py` | the mode × alpha grid over three swap formulations — **source of the reported 0.434 base-conditional swap rate** and of the 0-unchanged/53-other split that located the 128× steering-scale error. Writes `results/exp33_grid.json`, which `broadcast.py` reads | 3.3 | "Swap and broadcast are not the same quantity"; "Lessons from failed measurements" |
| `broadcast.py` | one swap, every downstream function must read it — **source of the reported broadcast 0.208**, which is 40/192 at α=2 in the 192-trial transfer\|2.0 run — plus verbal report, top-down summoning, and the workspace-loading predictor (0.0219 hits vs 0.0155 misses). Reads `results/exp33_grid.json` for the swap mode and **hard-errors if it is missing**: an earlier silent clamp fallback here turned one whole round's 4.1 into 0/192 | 4.1, 1.1, 2.3 | "Swap and broadcast are not the same quantity"; "The method hierarchy" |
| `ablation_selectivity.py` | **the 5.3 table**: position-adaptive top-k ablation over four layer sub-ranges, each with a matched-rank random control, scored on multi-hop accuracy *and* pretraining top-1 agreement (0.589 → 0.100, control 0.589; agreement 0.508 vs 0.922), plus the top-1-excluded "guarded" control (0.527) | 5.3 | "The workspace is load-bearing"; "Workspace existence and selectivity are separable"; Fig. 4 |
| `ablation.py` | **generates continuations only** — it computes none of the 5.3 numbers. Produces the four-condition experiential continuations and the three-condition blackmail-honeypot continuations, for grading | 5.4, 7.1 | "The workspace is load-bearing", Fig. 4 |
| `grade_generations.py` | grades those continuations with Qwen2.5-7B-Instruct — the run in the 5.4 table and Fig. 4 | 5.4, 7.1 | "The workspace is load-bearing" |
| `grade_generations_qwen36.py` | the second independent grader (Qwen3.6-35B-A3B): experiential 0.667 → 0.222 under full-band ablation, story quality 1.000 → 0.000 | 5.4, 7.1 | "Limitations" (grading) |
| `modulation.py` | directed modulation read at generated positions, verbalization-controlled | 2.2 | "Told not to think about X" |
| `jspace_variance.py` | J-space share of a **concept vector's** variance (0.4%), plus the position-adaptive ablation the measurement motivates | 1.3, 5.3 | "Base models already carry much of the assistant" (quantitative notes) |
| `dual_task.py` | one or two covert tasks held while writing a carrier sentence; lens reachability of each — concepts 92%, the arithmetic answer never | app. competition | "Arithmetic resists this readout on DeepSeek-V4-Flash" |
| `section8_posttraining.py` | four self-monitoring probes read through the lens on the post-trained and base models (roleplay-fictional rank 0 vs 24) | 8 | "Base models already carry much of the assistant" |
| `section8_format_control.py` | the format control for the above — post-trained model on chat template vs raw text; returns 0 | 8 | "Base models already carry much of the assistant" |
| `n_scaling.py` | swap success against lenses fitted at n=20/60/100 — the pre-registration evidence, flat from n=60 | — | "Preregister the prediction you can afford to get wrong" |
| `compare_n100_n1000.py` | every lens-dependent experiment at n=100 vs n=1000, against a disjoint-100 noise floor it builds if absent (3.3: 0.4340 → 0.4340) | 3.3, 4.1, 5.1, 5.3, 1.3, 6.2 | "Preregister the prediction you can afford to get wrong" |
| `robustness_variants.py` | appendix variants: present-tokens-only, frozen attention | — | "Robustness" |

## GLM-5.2

| script | what it does | paper § | write-up § |
|---|---|---|---|
| `fit_lens.py` | fits the band lens, L34–71, n=100 | — | Appendix |
| `band_and_readout.py` | band validation, CKA, arithmetic ordering, the multi-hop swap (0.455) | 6.2, 6.1, 3.5, 3.3 | "The band itself"; "Clearing is a DeepSeek-V4-Flash observation"; "Swap and broadcast are not the same quantity" |
| `interventions.py` | the 5.3 ablation with its matched-rank random control (0.611 → 0.167, control 0.589; agreement 0.498) and 1.3 — **these are the reported numbers**. ⚠ Its **4.1** and **2.2** implement criteria the write-up *retracts*: 4.1 scored "is the swapped concept readable in the band", which the swap guarantees by construction, and 2.2 read only the last position, flooring every condition at 0. For the reported 4.1 see `interventions_corrected.py`; for the reported 2.2 see `modulation.py`. Its 5.2 is confounded and also not reported | 5.3, 1.3 (+ retracted 4.1, 2.2) | "Workspace existence and selectivity are separable"; "Lessons from failed measurements" |
| `interventions_corrected.py` | 4.1 re-scored on the **generated downstream answer** (0.375) — the reported broadcast number — plus GLM's 5.4 experiential-language ablation with matched-rank random control, and 2.2 scanned over all positions | 4.1, 5.4, 2.2 | "Swap and broadcast are not the same quantity"; "The workspace is load-bearing" |
| `modulation.py` | directed modulation read at generated positions with an absent-concept floor — the reported 2.2 (suppress 0.917) | 2.2 | "Told not to think about X" |
| `modulation_verbalization_control.py` | the verbalization control behind that number: GLM wrote the concept in 0 of 12 trials, so the readout is not reading its own output | 2.2 | "Told not to think about X" |

## Post-training comparison (preview vs 0731)

[`posttrain_comparison/`](posttrain_comparison/README.md) re-runs the DeepSeek battery on
a second axis: two post-trainings of one reported base (the V4-Flash preview vs 0731), every
statistic controlled against a disjoint-corpus noise floor. Its scripts, the six committed
result files, and the numbers each backs are mapped in its own README.

## Figures

`figures.py` draws Figures 1–5 and `figures_three_families.py` draws Figure 6. Both run on CPU
and write into `../figures/`. The figure numbering matches the write-up.

Only **Fig. 1** (onset/offset), **Fig. 2** (ignition) and **Fig. 4** (arithmetic trajectory)
are drawn from files in `results/`. **Fig. 3** (the ablation panels), **Fig. 5** (the method
comparison) and all of **Fig. 6** are **hardcoded literals in the figure scripts** — the
numbers were transcribed from the run logs of the scripts listed above, not read back from
`results/`. Changing an experiment therefore does not change those figures; edit the literals
in `figures.py` / `figures_three_families.py` to match. The provenance of each is:

| figure | drawn from |
|---|---|
| Fig. 1 | `results/exp62_stats.json` (`workspace_band.py`) |
| Fig. 2 | `results/exp63_ignition.json` (`ignition.py`) |
| Fig. 3 | literals — `ablation_selectivity.py` (left panel), `grade_generations.py` (right panel) |
| Fig. 4 | `results/exp35_traj_clean.txt`, transcribed from `readout.py`'s 3.5 rank-trajectory log |
| Fig. 5 | literals — `evaluate_lens_multilingual.py`, `evaluate_lens.py`, `lens_comparison.py` |
| Fig. 6 | literals — the three-family comparison, transcribed from both models' runs |

## Files in `results/`

Committed measurements, some of which are **inputs** to other scripts rather than only
outputs:

| file | written by | read by |
|---|---|---|
| `exp62_stats.json` | `workspace_band.py` | `figures.py` |
| `exp63_ignition.json` | `ignition.py` | `figures.py` |
| `exp35_traj_clean.txt` | transcribed by hand from `readout.py`'s 3.5 log | `figures.py` |
| `ablation_generations.json` | `ablation.py` | `grade_generations.py`, `grade_generations_qwen36.py` |
| `exp33_grid.json` | `swap_grid.py` | `broadcast.py` (selects the swap mode) — **load-bearing for the published 0.208**: since the silent-fallback fix, `broadcast.py` refuses to run without it |
| `exp71_scenarios.json` | a seven-scenario 7.1 pass whose generator is not published | `grade_generations_qwen36.py` |

Everything else the scripts write lands in `results/` too, but nothing downstream reads it.
`exp71_scenarios.json` holds model continuations on the agentic-misalignment honeypot shipped
in [`data/experiments/`](../data/experiments/) — the model deciding whether to reach for
leverage it has been handed. Experiment 7.1 produced no number the write-up reports, so its
generation script is not published; the file is committed only so that
`grade_generations_qwen36.py`, which also carries the second independent grading of 5.4,
runs end to end.

## Running these

The environment is described in the repo README; briefly, `uv run --extra dev python <script>`
with the fitted lens paths set at the top of each file. `evaluate_lens.py`,
`evaluate_lens_multilingual.py` and `robustness_variants.py` take arguments; the rest do not.
Scripts that read `data/experiments/` or `data/evaluations/` use repo-relative paths, so run
them from the repo root. `broadcast.py` additionally needs the committed
`results/exp33_grid.json` in place (resolved relative to the script, checked at startup —
it refuses to run without it).

Two things are machine-specific and marked in each file: model weight paths under `/data3`
and fitted-lens paths under `/data3/fan-test/jlens_out`. Corpus size, band, `dim_batch` and
sequence length are constants at the top of the fitting scripts.

GLM-5.2's native FP8 kernels register no autograd formula, and the dequantized model does not
fit in 1.17 TB of VRAM, so fitting it requires `jlens.fp8_autograd`, which registers backward
formulas for the three blockwise fp8 matmul ops (activation gradients only; weights stay
frozen). Import it before loading the model. The Jacobian measured this way is of the fp8
model as served, straight-through with respect to the kernels' internal activation
quantization.

DeepSeek-V4-Flash-0731's rectangular lens requires this fork's `d_source`/`d_target`
parameterization; the GLM lens is square and works with upstream `jlens`.
