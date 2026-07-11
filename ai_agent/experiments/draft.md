---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2
summary: Test the old baseline lr=5e-4 combined with the champion's dropout=0.05 — checks whether lr=7e-4's failure was an isolated valley
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
Trial 04d1d849 (`lr=7e-4` + `dropout=0.05`) failed badly (delta -0.083, real
instability) — surprising since 7e-4 sits between the old baseline's 5e-4 and the
champion's 8e-4, both of which work reasonably well with dropout in some form
(5e-4 with dropout=0: 0.807; 8e-4 with dropout=0.05: 0.828, on average ~0.80-0.83).
This suggests either a genuine non-monotonic "valley" at 7e-4 specifically, or that
dropout=0.05 combines well ONLY with lr=8e-4 and poorly with anything else nearby
(including lower, not just higher, values). Testing `lr=5e-4` + `dropout=0.05`
(never tried — 5e-4 was only tested with dropout=0 as the old baseline) distinguishes
these: a good result would support "7e-4 is an isolated valley"; a bad result would
support "dropout=0.05 only works with lr=8e-4 specifically".

## Implementation
Single-field change relative to the champion: `lr: 8e-4 -> 5e-4`. `dropout_rate=0.05`
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