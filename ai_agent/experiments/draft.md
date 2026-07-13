---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2
summary: "Third replicate of patience=45 (0.8089, 0.8060, ?) to confirm the apparently much lower run-to-run variance vs patience=60's 4-point spread of 0.075"
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
patience=45's first two draws (0.8089, 0.8060) differ by only 0.0029, far tighter
than patience=60's 4-point spread (0.075). If this is a real, mechanistically-grounded
effect (shorter schedules compound less stochastic noise from batch_size=1 training),
a third draw should also land close to this tight ~0.806-0.809 band. If it instead
lands far outside (e.g. near patience=60's lower tail of ~0.74-0.78), the apparent
tightness of the first two points was itself a coincidence at n=2, and no reliable
variance difference between the two schedule lengths can be claimed.

## Implementation
Single-field change in configs/autoencoder.yaml: patience 60 -> 45 (patience_scheduler
= 9), on top of lr=6e-4, noise_std=0.0001. weight_decay=0, dropout_rate=0 unchanged.
No architecture change. Third replicate of the same config as 69832c74/59547f27.

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
