---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2SELateEnc3Only
summary: "Complete the SE-placement ablation grid: se3 only (remove se4 too), isolating whether enc3's gate alone captures the champion's benefit"
parent: 761cab78

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
Seven consecutive non-SE trials this campaign (`1ee5e03c` through `498a57b2`: bottleneck capacity,
normalization, residual, a new SE location, two dilation locations, and both directions of an `lr`
sweep) have all failed to beat the champion — returning to the one mechanism with a clean record
(SE channel gating) to finish characterizing it before trying anything else new. The SE-placement
ablation grid so far: neither (`ff5882ad` BASELINE, 0.6667), full se1-se4 (`a581f44e` CHAMPION,
0.6917), se3+se4 (`761cab78` CHAMPION, 0.7000), se4 only (`09415e52` FAILURE, 0.675). The one
untested cell is se3 only (drop se4, keep se3) — `761cab78` vs `09415e52` showed se3's gate matters
more than se4's alone, but that comparison can't tell whether se3 alone already captures most/all of
the champion's benefit (se4 close to redundant, and this becomes a simpler, marginally smaller
champion candidate — 512 fewer params) or whether se3 and se4 need each other (an interaction effect,
in which case this should underperform both `761cab78` and `09415e52`). Predicting this either
matches the champion (se4 redundant) or sits between `09415e52` and `761cab78` (partial interaction).
No encoder-to-decoder path — respects the no-skip-connections rule.

## Implementation
New class `AutoEncoder3D_AsymResSeparableV2_SELateEnc3Only` in `src/models/ae_models.py` (model_name
`AE3dAsymResSeparableV2SELateEnc3Only`), copied from `AutoEncoder3D_AsymResSeparableV2_SELate` (parent
`761cab78`) with exactly one change: `se4` (and its call in `encode`) removed, keeping only `se3`.
Everything else (enc4 dilation=1, `bottleneck_conv` unchanged, hyperparameters `lr=1e-4`,
`weight_decay=0`, `dropout_rate=0`, `noise_std=0`, `patience=20`) identical to the champion. Verified
the new class builds, forward-passes, and produces the expected (1,20) latent: param count 1,128,969
vs champion `761cab78`'s 1,129,481 (delta -512 = exactly `SEBlock3D(64, reduction=16)`'s params,
64*4 + 4*64, confirming only se4 was removed).

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