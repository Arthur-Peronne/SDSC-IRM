---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2
summary: "Test small input-space denoising (noise_std 0.0 -> 0.0001) on top of the champion's lr=6e-4, the one regularization mechanism not yet tried (dropout and weight_decay both failed)"
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
Both dropout (0.05, weight-space stochastic) and weight_decay (1e-6, weight-space
smooth) failed on top of the champion, the former badly, the latter negligibly —
consistent with the champion's already-tight train/val gap (0.045) leaving little
room for regularization to help. noise_std is mechanistically different: it perturbs
the INPUT rather than the weights/activations, forcing the encoder to learn features
robust to small voxel-intensity noise rather than penalizing capacity directly. I
test a small value (0.0001) — deliberately far below the AEoptuna reference of 0.002,
since a similar denoising magnitude caused a catastrophic failure (R2 -0.185) in the
dim=240 campaign at its (much lower) baseline lr. Given the pattern so far, I predict
another FAILURE (the tight gap argument applies here too), but a distinct mechanism
is still worth the one test to complete the regularization sweep before moving to
patience.

## Implementation
Single-field change in configs/autoencoder.yaml: noise_std 0.0 -> 0.0001, on top of
the champion's lr=6e-4. weight_decay=0, dropout_rate=0 (both reverted), patience=30
unchanged. No architecture change.

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
