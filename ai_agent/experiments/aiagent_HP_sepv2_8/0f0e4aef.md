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
Best epoch 57, early stopping at 107. Val loss (0.000599) normal, no
instability — a clean final run.

## Conclusion
**Campaign's final replicate series for lr=7e-4, n=13: mean=0.7957,
std≈0.0097.** This is the campaign's headline, well-supported recommendation:
`lr=7e-4, weight_decay=0, dropout_rate=0, noise_std=0, patience=50` at
latent_dimensions=8, diverging from dim=60's optimum (lr=6e-4) in exactly one
axis, with a mean improvement of ~0.014 over the dim=60-transferred baseline
(0.7817 at lr=6e-4, n=6) and dramatically better reliability than the riskier
lr=7.5e-4 alternative (n=3, bimodal, std=0.057). This was the 40th and final
trial of the campaign budget (`max_trials=40`).