---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2SE4Only      # the AE architecture trained (e.g. AE3dDilatedAttention). Descriptive, NOT the id.
summary: "Further ablation: keep champion's SE only on enc4 (remove se3 too)"         # one-line description of the change (becomes the CSV modification_description)
parent: 761cab78         # lineage: `id` of the trial whose CODE this one branched FROM. null for roots.

# ---- driver-written (leave null; the driver overwrites at lock/result) ----
id: null              # short sha of commit 1 == the frozen input == this trial's identity
status: draft         # draft -> completed | failed          (lifecycle, lowercase)
verdict: null         # BASELINE | CHAMPION | CANDIDATE | FAILURE   (judgement, UPPERCASE)
created_at: null
metric:
  primary: {name: avg_validation_R2_mean, value: null, direction: maximize}
---

# Trial <id> — <model_name> — <verdict>

<!-- ===== written BEFORE the run (agent) ===== -->

## Hypothesis
Further ablation of the champion `AE3dAsymResSeparableV2SELate` (trial 761cab78, which showed
se1/se2 net harmful and se3+se4 net helpful vs. full SE). This trial removes `se3` too, keeping SE
only on `enc4` (64 channels, immediately before `bottleneck_conv`). Mechanism question: is `se3` (32
channels, still one pooling stage before the bottleneck) itself contributing meaningfully, or is the
champion's whole gain carried by `se4` alone? If `se4`-only matches or beats 761cab78, the effect is
concentrated at the very last stage and `se3` can be dropped for free; if it underperforms,
`se3` is genuinely contributing and 761cab78 (both gates) is the better design. Same hyperparameters
as all prior trials (`lr=1e-4`, `weight_decay=0`, `dropout_rate=0`, `noise_std=0`, `patience=20`).

## Implementation
New class `AutoEncoder3D_AsymResSeparableV2_SE4Only` in `src/models/ae_models.py` (model_name
`AE3dAsymResSeparableV2SE4Only`), copied from `AutoEncoder3D_AsymResSeparableV2_SELate` (current
champion) with `se3` removed — `enc3` now feeds directly into `z_pool3` with no gating. `se4` kept
exactly as in the champion. Decoder unchanged. Same-stage gating only, no encoder-to-decoder path —
respects the no-skip-connections rule in `program.md`. Verified the new class builds, forward-passes,
and produces a (1,20) latent before launching the trial. This completes the SE-placement ablation
series (all4 -> late-only -> last-only); the next trial after this one will diversify to a
different architectural family regardless of this result, to avoid over-indexing on
diminishing-returns variants of the same idea.

<!-- ===== written AFTER the run ===== -->

## Results
<!-- Filled automatically by the driver — leave empty. It writes, for a completed trial:
     per-run metric values (by repeat axis), the aggregated primary metric,
     delta_vs_champion (display only), the also_log means, and the MLflow run ids.
     For a mechanically failed trial it writes the failure reason instead. -->

## Training Dynamics
<!-- Agent, after the run: stability, convergence speed, spikes, plateau, early stopping. -->

## Conclusion
<!-- Agent, after the run: did the hypothesis hold? Mechanistic explanation of why it
     worked or failed — not just the numbers. -->