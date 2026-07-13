---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2
summary: "Push the stabilizing combo further — weight_decay=1e-5 (10x more) + noise_std=0.0002, to test whether more weight_decay continues rescuing the noise_std instability or whether 1e-6 was already the useful amount"
parent: bc589070

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
weight_decay=1e-6 + noise_std=0.0002 substantially stabilized what was a chaotic
FAILURE alone (R2 0.7252 -> 0.7854, val R2 std 0.1282 -> 0.0663). If the
stabilization mechanism (weight shrinkage damping gradient-noise interaction) scales
with weight_decay's magnitude, a 10x larger weight_decay (1e-5) — which alone is
only mildly negative (-0.008) — might stabilize the noise_std=0.0002 config even
further, potentially closing more of the gap to the champion. If instead 1e-6 was
already sufficient and more weight_decay just adds its own (small) capacity cost on
top without further stabilization benefit, the result should land similarly to or
slightly worse than 75165602's 0.7854.

## Implementation
Two-field change in configs/autoencoder.yaml: weight_decay 0.0 -> 1e-5 AND noise_std
0.0001 -> 0.0002, simultaneously, on top of the champion's lr=6e-4, patience=50
(patience left untouched). dropout_rate=0 unchanged. No architecture change.

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
