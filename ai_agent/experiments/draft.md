---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2
summary: "Combination test — weight_decay=1e-6 stacked on noise_std=0.0002 (which FAILED badly alone, -0.075, in a chaotic/unstable way) to test whether a smooth weight-space penalty can stabilize the noise_std cliff region"
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
noise_std=0.0002 alone FAILED badly (-0.075) with a non-monotonic, chaotic-looking
degradation — worse than noise_std=0.0003, suggesting instability rather than a
smooth dose-response. weight_decay is a smooth, deterministic regularizer that in
isolation is only mildly negative. I test whether adding a small weight_decay to the
unstable noise_std=0.0002 config changes the outcome: if weight_decay's smoothing
effect (shrinking weights, damping large gradient-driven updates) counteracts some
of the instability that made 0.0002 chaotic, the result could land closer to (or
even above) the champion, unlike anything single-axis testing showed; if the
instability is dominated by the input noise itself regardless of weight
regularization, this will fail similarly to noise_std=0.0002 alone.

## Implementation
Two-field change in configs/autoencoder.yaml: weight_decay 0.0 -> 1e-6 AND noise_std
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
