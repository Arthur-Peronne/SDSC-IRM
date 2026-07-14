---
model_name: AE3dAsymResSeparableV2
summary: 'lr=7e-4 replicate 9/N: continuing to firm up the final recommendation''s
  confidence interval with remaining budget'
parent: 71508734
id: efccddf2
status: completed
verdict: FAILURE
created_at: '2026-07-14T01:04:25+00:00'
metric:
  primary:
    name: avg_validation_R2_mean
    value: 0.792633
    direction: maximize
---

# Trial efccddf2 — AE3dAsymResSeparableV2 — FAILURE

<!-- ===== written BEFORE the run (agent) ===== -->

## Hypothesis
Continuing the lr=7e-4 replicate series (n=8: mean 0.7982, std≈0.0079). Pure
variance-reduction continuation, no new hypothesis.

## Implementation
No change to `configs/autoencoder.yaml`: lr=7e-4, weight_decay=0.0,
dropout_rate=0.0, noise_std=0.0, patience=50, latent_dimensions=8, seed=0.

<!-- ===== written AFTER the run ===== -->

## Results
- **validation_R2_mean per run:** e24c72a7: 0.792633
- **avg_validation_R2_mean:** 0.792633
- **delta_vs_champion** (display only): -0.020869
- **validation_MSE_mean** (mean, non-decisional): 161.071091
- **MLflow Run IDs:** e24c72a7562b429eb1491591cf8a0056

## Training Dynamics
<!-- Agent, after the run: stability, convergence speed, spikes, plateau, early stopping. -->

## Conclusion
<!-- Agent, after the run: did the hypothesis hold? Mechanistic explanation of why it
     worked or failed — not just the numbers. -->