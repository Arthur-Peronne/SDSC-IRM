---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2
summary: Stack a very small noise_std (0.0 -> 0.0001) on top of the champion's lr=8e-4 + dropout=0.05 — last untested combination
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
`noise_std` is the one opened HP never combined with the champion's `lr=8e-4` +
`dropout=0.05`. Trial de0f5947 showed stacking `weight_decay` was redundant
(neutral), and trial 571f33ab showed stacking `patience` was also redundant —
suggesting the "regularization budget" at this operating point is largely used by
dropout already. `noise_std` acts on inputs rather than weights, a distinct
mechanism, so it is not guaranteed to be redundant the same way. I use a very small
value (0.0001, half of trial 4bd7e912's already-mild-and-neutral 0.0002 on the old
baseline) to avoid the catastrophic instability seen at 0.002. Given the noise-floor
finding from trials 59ff727f/f5873a7c, I do not expect to reliably resolve a small
effect here — this trial is mainly for completeness of the opened-HP combination
space before closing out the campaign.

## Implementation
Single-field addition relative to the champion: `noise_std: 0.0 -> 0.0001`. `lr=8e-4`
and `dropout_rate=0.05` (the champion's values) unchanged. `weight_decay`, `patience`
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