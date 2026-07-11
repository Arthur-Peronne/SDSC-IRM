---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2
summary: Stack weight_decay=1e-6 on lr=5e-4 + dropout=0.05 — the "other" good lr point, never combined with a second regularizer
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
`weight_decay=1e-6` stacked on the champion's `lr=8e-4`+`dropout=0.05` was neutral
(trial de0f5947). All weight_decay/noise_std stacking so far has only been tried at
lr=8e-4. Trial 11530c52 established `lr=5e-4`+`dropout=0.05` as a second good,
stable point (0.808). This tests whether a tiny weight_decay addition behaves the
same way (neutral) at this different lr, completing the regularizer-stacking
picture across both known-good lr values rather than only the champion's.

## Implementation
Two-field change relative to the champion: `lr: 8e-4 -> 5e-4` AND
`weight_decay: 0.0 -> 1e-6`. `dropout_rate=0.05` unchanged. `noise_std`, `patience`
unchanged from the baseline defaults. No architectural change.

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