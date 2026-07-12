---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2
summary: "Test patience increase (30 -> 45) on top of the champion, the last opened HP not yet touched, to check whether more room after the plateau ever finds a better epoch"
parent: 319dacea

# ---- driver-written (leave null; the driver overwrites at lock/result) ----
id: null
status: draft
verdict: null
created_at: null
metric:
  primary: {name: avg_validation_R2_mean, value: null, direction: maximize}
---

# Trial <id> — <model_name> — <verdict>

<!-- ===== written BEFORE the run (agent) ===== -->

## Hypothesis
This is the last of the 5 opened HPs not yet tested since the baseline. In the
champion (319dacea), the plateau scheduler had already crushed lr to 1.87e-05 (near
frozen) by the time patience=30 expired at epoch 75, best epoch 45 — the last ~15-20
epochs before stopping showed no real movement. This suggests patience=30 is not
cutting off genuine ongoing improvement, so I predict patience=45 will most likely
be neutral-to-mildly-negative (matching the dim=240 campaign's own patience-increase
test, which also failed): training simply continues past the point of useful
learning-rate-driven change, wasting budget without the checkpoint-selection (best
val loss so far) changing, or occasionally picking a spuriously-lower noisy val
checkpoint late in a frozen-lr regime that does not truly generalize better.

## Implementation
Single-field change in configs/autoencoder.yaml: patience 30 -> 45, on top of the
champion's lr=6e-4, noise_std=0.0001. weight_decay=0, dropout_rate=0 unchanged. No
architecture change.

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
