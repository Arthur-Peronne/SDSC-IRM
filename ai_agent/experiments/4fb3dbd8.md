---
model_name: AE3dAsymResSeparableV2
summary: 'lr=7e-4 replicate 11/N: final trials of budget, tightening the recommendation''s
  confidence interval'
parent: 71508734
id: 4fb3dbd8
status: completed
verdict: FAILURE
created_at: '2026-07-14T01:42:12+00:00'
metric:
  primary:
    name: avg_validation_R2_mean
    value: 0.778445
    direction: maximize
---

# Trial 4fb3dbd8 — AE3dAsymResSeparableV2 — FAILURE

<!-- ===== written BEFORE the run (agent) ===== -->

## Hypothesis
Continuing the lr=7e-4 replicate series (n=10: mean=0.7964). 3 trials of budget
remain; no new hypothesis, pure confidence-tightening.

## Implementation
No change to `configs/autoencoder.yaml`: lr=7e-4, weight_decay=0.0,
dropout_rate=0.0, noise_std=0.0, patience=50, latent_dimensions=8, seed=0.

<!-- ===== written AFTER the run ===== -->

## Results
- **validation_R2_mean per run:** adde968d: 0.778445
- **avg_validation_R2_mean:** 0.778445
- **delta_vs_champion** (display only): -0.035057
- **validation_MSE_mean** (mean, non-decisional): 168.416702
- **MLflow Run IDs:** adde968d676b464ca0b024aa4778ccd3

## Training Dynamics
<!-- Agent, after the run: stability, convergence speed, spikes, plateau, early stopping. -->

## Conclusion
<!-- Agent, after the run: did the hypothesis hold? Mechanistic explanation of why it
     worked or failed — not just the numbers. -->