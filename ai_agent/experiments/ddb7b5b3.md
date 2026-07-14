---
model_name: AE3dAsymResSeparableV2
summary: 'lr=7e-4 replicate 8/N: continuing to firm up the final recommendation''s
  confidence interval with remaining budget'
parent: 71508734
id: ddb7b5b3
status: completed
verdict: FAILURE
created_at: '2026-07-14T00:42:18+00:00'
metric:
  primary:
    name: avg_validation_R2_mean
    value: 0.794724
    direction: maximize
---

# Trial ddb7b5b3 — AE3dAsymResSeparableV2 — FAILURE

<!-- ===== written BEFORE the run (agent) ===== -->

## Hypothesis
Continuing the lr=7e-4 replicate series (n=7: mean 0.7987, std≈0.0084). Pure
variance-reduction continuation with the remaining budget, no new hypothesis.

## Implementation
No change to `configs/autoencoder.yaml`: lr=7e-4, weight_decay=0.0,
dropout_rate=0.0, noise_std=0.0, patience=50, latent_dimensions=8, seed=0.

<!-- ===== written AFTER the run ===== -->

## Results
- **validation_R2_mean per run:** a5dc685d: 0.794724
- **avg_validation_R2_mean:** 0.794724
- **delta_vs_champion** (display only): -0.018778
- **validation_MSE_mean** (mean, non-decisional): 160.146164
- **MLflow Run IDs:** a5dc685df8f444e682b2a136807c2f3d

## Training Dynamics
<!-- Agent, after the run: stability, convergence speed, spikes, plateau, early stopping. -->

## Conclusion
<!-- Agent, after the run: did the hypothesis hold? Mechanistic explanation of why it
     worked or failed — not just the numbers. -->