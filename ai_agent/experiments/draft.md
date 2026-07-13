---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2
summary: "Fourth replicate of patience=45 to test whether its apparent tight variance (3 points, std=0.008) survives a larger sample, now that patience=50's 2 points already showed wide (0.035) spread"
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
patience=45's first 3 draws (0.8089, 0.8060, 0.7913) were unusually tight
(std≈0.008) compared to patience=50 (2 draws, 0.8148/0.7796, spread 0.035) and
patience=60 (5 draws, std≈0.035). A 4th patience=45 draw is the key test: if it also
lands in the tight ~0.79-0.81 band, the tightness is likely real and specific to this
value; if it lands far outside (e.g. near 0.74 or below 0.78), the tight cluster was
itself a small-sample coincidence and patience=45 is no more reproducible than its
neighbors — all patience values in this range would then plausibly share the same
wide underlying noise distribution.

## Implementation
Single-field change in configs/autoencoder.yaml: patience 50 -> 45 (patience_scheduler
= 9), on top of lr=6e-4, noise_std=0.0001. weight_decay=0, dropout_rate=0 unchanged.
No architecture change. Fourth replicate of 69832c74/59547f27/3fbb7661's config.

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
