---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2
summary: Stack a very small weight_decay (0.0 -> 1e-6) on top of the champion's lr=8e-4 + dropout=0.05
parent: 3e07b08d

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
The dropout/lr interaction (trial 3e07b08d) taught a general lesson: regularization
that failed at a naively-large magnitude (dropout=0.15, weight_decay=1e-5, both
picked as "moderate" values without reference to this specific model) can still help
at a much smaller dose once combined with the right lr. `weight_decay=1e-5` alone
failed clearly (trial 5fb49bb5, delta -0.029) at the old lr=5e-4. I now test
`weight_decay=1e-6` — 10x smaller — stacked on the current champion's `lr=8e-4` +
`dropout=0.05`, to see whether a second, much gentler regularizer adds a further
increment the way dropout did, or whether the champion's combination is already
using up the available "regularization budget" (in which case a third regularizer,
however small, should be redundant or mildly negative rather than additive).

## Implementation
Single-field addition relative to the champion: `weight_decay: 0.0 -> 1e-6`.
`lr=8e-4` and `dropout_rate=0.05` (the champion's values) unchanged. `noise_std`,
`patience` unchanged from the baseline defaults. No architectural change.

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