---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2
summary: "Test patience=50 (between 45's tight-variance cluster and 60's wide-variance cluster) to locate where the apparent variance transition happens"
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
patience=45 (3 draws, std≈0.008) and patience=60 (5 draws, std≈0.035) show
substantially different run-to-run variance. This single point at patience=50
(patience_scheduler=10) cannot establish a second variance estimate on its own, but
its point value gives a first hint: if it lands within patience=45's tight ~0.79-0.81
band, that suggests the transition to higher variance happens closer to 60; if it
already shows signs of the wider patience=60-like spread (relative to what a longer
series would reveal), it suggests the variance increase begins earlier, closer to
50. This is exploratory rather than a decisive test at n=1.

## Implementation
Single-field change in configs/autoencoder.yaml: patience 60 -> 50 (patience_scheduler
auto-derived as patience//5 = 10), on top of lr=6e-4, noise_std=0.0001. weight_decay=0,
dropout_rate=0 unchanged. No architecture change.

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
