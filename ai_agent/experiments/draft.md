---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2
summary: "Last lr step (6e-4 -> 8e-4), matching the dim=240 champion exactly, to close the lr axis given sharply diminishing gains and the first visible raw-lr noise at 6e-4"
parent: 97d513a3

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
The last three lr increases won with sharply diminishing returns (+0.096, +0.040,
+0.010) and 6e-4 showed the first visible raw-lr noise (val loss jumping more within
an epoch window than at 3e-4), while still no divergence/NaN. This is the profile of
approaching a ceiling. I test lr=8e-4 (matching the dim=240 campaign's champion
exactly) as the deliberate last step on this axis: if the trend continues even
slightly, there is still headroom and I would consider one more step; if it plateaus,
regresses, or shows outright instability, that firmly closes the lr axis at 6e-4 (the
best seen so far) and moves exploration to weight_decay / dropout_rate / noise_std /
patience, none of which have been touched since the neutral baseline.

## Implementation
Single-field change in configs/autoencoder.yaml: lr 6e-4 -> 8e-4. All other opened
HPs unchanged (weight_decay=0, dropout_rate=0, noise_std=0, patience=30). No
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
