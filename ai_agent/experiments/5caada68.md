---
model_name: AE3dAsymResSeparableV2
summary: 'Baseline replicate 6/6 (noise_std=0.0001): completing the matched n=6 sample
  against the champion'
parent: 0d2e0fa2
id: 5caada68
status: completed
verdict: FAILURE
created_at: '2026-07-13T20:30:05+00:00'
metric:
  primary:
    name: avg_validation_R2_mean
    value: 0.759396
    direction: maximize
---

# Trial 5caada68 — AE3dAsymResSeparableV2 — FAILURE

<!-- ===== written BEFORE the run (agent) ===== -->

## Hypothesis
Final baseline replicate to reach the matched n=6 (mean so far 0.7832,
std≈0.0228, essentially tied with the champion's 0.7817/0.0238). Completing this
closes the noise_std comparison with equal sample sizes on both sides.

## Implementation
`configs/autoencoder.yaml`: `noise_std: 0.0001` (baseline's value, unchanged),
lr=6e-4, weight_decay=0.0, dropout_rate=0.0, patience=50, latent_dimensions=8,
seed=0.

<!-- ===== written AFTER the run ===== -->

## Results
- **validation_R2_mean per run:** 630adaa7: 0.759396
- **avg_validation_R2_mean:** 0.759396
- **delta_vs_champion** (display only): -0.046214
- **validation_MSE_mean** (mean, non-decisional): 176.816010
- **MLflow Run IDs:** 630adaa79a864ac7a0ec9e7c5e66ff83

## Training Dynamics
<!-- Agent, after the run: stability, convergence speed, spikes, plateau, early stopping. -->

## Conclusion
<!-- Agent, after the run: did the hypothesis hold? Mechanistic explanation of why it
     worked or failed — not just the numbers. -->