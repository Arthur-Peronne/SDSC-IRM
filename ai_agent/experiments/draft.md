---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2
summary: Explore neighborhood of new champion 3e07b08d — increase dropout (0.05 -> 0.08) at lr=8e-4, to test if more regularization still helps
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
The new champion 3e07b08d (`lr=8e-4`, `dropout_rate=0.05`) confirmed that dropout
helps once paired with the faster-converging higher lr, unlike dropout alone on the
old baseline. Its train/val R2 gap (0.870 vs 0.828 = 0.042) is narrower than the old
baseline's (0.052) but not fully closed, so there may be room for a bit more
regularization. I set `dropout_rate: 0.05 -> 0.08` (still far below the failed 0.15)
at the same `lr=8e-4`, predicting either a further improvement (if 0.05 was not yet
the ceiling) or a plateau/slight regression (if 0.05 is already near-optimal for this
lr) — genuinely testing which side of the new champion's dropout value we are on.

## Implementation
Single-field change relative to the new champion: `dropout_rate: 0.05 -> 0.08`.
`lr=8e-4` (already the champion's value) unchanged. `weight_decay`, `noise_std`,
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