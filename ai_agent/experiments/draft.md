---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2
summary: Add mild input-space denoising (noise_std 0.0 -> 0.002), an axis not yet tried after 3 FAILUREs on dropout/weight_decay/lr
parent: 185cf97f

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
Trials 6a8a38d6 (dropout), 5fb49bb5 (weight_decay), and e44c7fa6 (lower lr) all FAILED,
via three different mechanisms, all of which act on the model's weights or the
optimization trajectory. `noise_std` (input-space Gaussian corruption, denoising-AE
style) is mechanistically different: it does not restrain the weights or the
optimizer, it changes what the model is trained to reconstruct FROM — a corrupted
input mapped back to a clean target. This encourages the encoder to learn features
robust to small perturbations rather than exact-pixel fitting, which could reduce
val-time sensitivity to the natural patient-to-patient variability without touching
model capacity or step size (the two things already shown to hurt when altered). I use
`noise_std=0.002`, the magnitude already validated by the project's own Optuna search
(AEoptuna value noted in the YAML comment) for a comparable setting. I predict
`avg_validation_R2_mean` improves above 0.8075.

## Implementation
Single-field change in `configs/autoencoder.yaml`: `noise_std: 0.0 -> 0.002`. All other
hyperparameters unchanged from the baseline (lr=5e-4, weight_decay=0.0,
dropout_rate=0.0, patience=30). No architectural change.

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