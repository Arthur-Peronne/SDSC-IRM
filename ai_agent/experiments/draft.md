---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2
summary: "Interaction check — retest lr=8e-4 (which FAILED under patience=30/patience_scheduler=6) now under the champion's patience=60/patience_scheduler=12, to see if the longer schedule changes the outcome"
parent: bed745a0

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
All 5 opened HPs have now been individually mapped from a moving champion, but never
jointly re-checked: lr=8e-4 was rejected under patience=30 (patience_scheduler=6) due
to raw-lr instability (an epoch-7 spike, noisier descent, worse/slower convergence
than 6e-4). Since then, patience grew to 60 (patience_scheduler=12), doubling how long
the model sits at its initial lr before the first decay. I predict this interaction
makes lr=8e-4 fail even more clearly than before: the same instability mechanism
(large single-sample gradient steps at batch_size=1) now gets twice as many epochs to
compound before any cooling arrives, rather than being tamed by it. A confirmed
FAILURE here would show the lr and patience optima are independent (safe to combine
each axis's individually-found best value); a surprise CHAMPION would mean the axes
interact and 6e-4 was only locally optimal under the old, shorter schedule.

## Implementation
Single-field change in configs/autoencoder.yaml: lr 6e-4 -> 8e-4, on top of the
champion's patience=60, noise_std=0.0001. weight_decay=0, dropout_rate=0 unchanged.
No architecture change.

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
