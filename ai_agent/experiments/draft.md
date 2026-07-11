---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2
summary: Check the other side of the champion's dropout value — dropout_rate 0.05 -> 0.03, at lr=8e-4
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
Trial 3faa0f66 showed `dropout_rate=0.08` (at `lr=8e-4`) clearly overshoots (delta
-0.072, real instability). This trial checks the other direction: `dropout_rate=0.03`,
below the champion's 0.05. If 0.05 sits past the true optimum on the way up from 0.0
(old baseline, no dropout) to 0.08 (too much), a smaller value like 0.03 might do
better still. If instead 0.05 is close to a genuine local optimum, 0.03 should come in
below the champion but likely still above the un-regularized baseline (0.807),
consistent with dropout being generally beneficial in this lr regime, just tuned to
a particular best magnitude.

## Implementation
Single-field change relative to the champion: `dropout_rate: 0.05 -> 0.03`. `lr=8e-4`
unchanged. `weight_decay`, `noise_std`, `patience` unchanged from the baseline
defaults. No architectural change.

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