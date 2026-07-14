---
model_name: AE3dAsymResSeparableV2
summary: 'lr=7e-4 replicate 13/N: final trial of the campaign budget (max_trials=40)'
parent: 71508734
id: 0f0e4aef
status: completed
verdict: FAILURE
created_at: '2026-07-14T02:20:26+00:00'
metric:
  primary:
    name: avg_validation_R2_mean
    value: 0.795348
    direction: maximize
---

# Trial 0f0e4aef — AE3dAsymResSeparableV2 — FAILURE

<!-- ===== written BEFORE the run (agent) ===== -->

## Hypothesis
Final trial of the campaign's 40-trial budget. Continuing the lr=7e-4 replicate
series (n=12: mean=0.7958, std≈0.0097) one last time before the driver reaches
max_trials and the campaign concludes. No new hypothesis.

## Implementation
No change to `configs/autoencoder.yaml`: lr=7e-4, weight_decay=0.0,
dropout_rate=0.0, noise_std=0.0, patience=50, latent_dimensions=8, seed=0.

<!-- ===== written AFTER the run ===== -->

## Results
- **validation_R2_mean per run:** 63948d00: 0.795348
- **avg_validation_R2_mean:** 0.795348
- **delta_vs_champion** (display only): -0.018154
- **validation_MSE_mean** (mean, non-decisional): 157.831314
- **MLflow Run IDs:** 63948d00fe6f49889cb9d284fe3491f3

## Training Dynamics
<!-- Agent, after the run: stability, convergence speed, spikes, plateau, early stopping. -->

## Conclusion
<!-- Agent, after the run: did the hypothesis hold? Mechanistic explanation of why it
     worked or failed — not just the numbers. -->