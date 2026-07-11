---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2
summary: Test dropout=0.08 at lr=5e-4 (never tried at this lr) — checks whether the dropout ceiling found at lr=8e-4 transfers to a different lr
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
Trial 11530c52 established that `lr=5e-4` + `dropout=0.05` trains well (0.808). At
`lr=8e-4`, `dropout=0.08` failed clearly (trial 3faa0f66, delta -0.072, real
instability). This trial tests `dropout=0.08` at the different lr=5e-4 to see
whether the dropout ceiling is a property of the architecture/data (transfers across
lr) or specific to the lr=8e-4 regime (in which case 0.08 might be fine at 5e-4,
similar to how the "bad valley" at lr=7e-4 did not generalize from a naive
expectation).

## Implementation
Two-field change relative to the champion: `lr: 8e-4 -> 5e-4` AND
`dropout_rate: 0.05 -> 0.08`. `weight_decay`, `noise_std`, `patience` unchanged from
the baseline defaults. No architectural change.

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