---
model_name: AE3dAsymResSeparableV2
summary: Test small input-space denoising (noise_std 0.0 -> 0.0001) on top of the
  champion's lr=6e-4, the one regularization mechanism not yet tried (dropout and
  weight_decay both failed)
parent: 97d513a3
id: 319dacea
status: completed
verdict: CHAMPION
created_at: '2026-07-12T20:44:49+00:00'
metric:
  primary:
    name: avg_validation_R2_mean
    value: 0.79996
    direction: maximize
---

# Trial 319dacea — AE3dAsymResSeparableV2 — CHAMPION

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
- **validation_R2_mean per run:** c8052396: 0.799960
- **avg_validation_R2_mean:** 0.799960
- **delta_vs_champion** (display only): +0.002720
- **validation_MSE_mean** (mean, non-decisional): 150.229004
- **MLflow Run IDs:** c80523969edd4e969b6dfbe0d634bb60

## Training Dynamics
<!-- Agent, after the run: stability, convergence speed, spikes, plateau, early stopping. -->

## Conclusion
<!-- Agent, after the run: did the hypothesis hold? Mechanistic explanation of why it
     worked or failed — not just the numbers. -->
