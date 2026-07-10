---
model_name: AE3dAsymResSeparableV2
summary: 'lr 5e-5 -> 5e-4 (10x): faster per-epoch convergence under the fixed 5-epoch
  budget'
parent: 21acba9e
id: e3b6e5ff
status: completed
verdict: CHAMPION
created_at: '2026-07-10T18:53:17+00:00'
metric:
  primary:
    name: avg_validation_R2_mean
    value: 0.5212
    direction: maximize
---

# Trial e3b6e5ff — AE3dAsymResSeparableV2 — CHAMPION

<!-- ===== written BEFORE the run (agent) ===== -->

## Hypothesis
In the BASELINE (21acba9e), val loss at all 3 latent dims was still descending steeply
at epoch 5/5 (e.g. dim=8: 0.072 -> 0.0029, still improving, no plateau) — the campaign's
fixed `n_epochs: 5` cuts training off long before convergence, not because the model has
converged. Under a hard epoch-budget constraint (n_epochs is frozen, not in the mutable
5), the direct lever to make more progress per epoch is a higher learning rate. I
increase `lr` from 5e-5 to 5e-4 (10x, one log-step), all else unchanged. I predict this
raises `avg_validation_R2_mean` above the BASELINE's -0.5428, because faster convergence
per step should let the model reach a lower loss / higher R² within the same 5-epoch
budget, mechanistically compensating for the truncated schedule (rather than fixing an
optimization pathology — none was observed in the BASELINE curves).

## Implementation
`configs/autoencoder.yaml`: `lr: 5e-5` -> `lr: 5e-4`. No other field touched
(`weight_decay`, `dropout_rate`, `noise_std`, `patience` unchanged from BASELINE;
architecture, data, split, `n_epochs` all frozen/untouched).

<!-- ===== written AFTER the run ===== -->

## Results
- **validation_R2_mean per run:** 8: 0.525491 | 40: 0.604250 | 200: 0.433860
- **avg_validation_R2_mean:** 0.521200
- **delta_vs_champion** (display only): +1.063956
- **validation_MSE_mean** (mean, non-decisional): 295.094330
- **MLflow Run IDs:** 7e623cc6274349728bbc50da210a4b50 677f9ee35b74423c9bc5f4d051333531 dc8497f0f3c240ff8f0e63c7d1e8c159

## Training Dynamics
<!-- Agent, after the run: stability, convergence speed, spikes, plateau, early stopping. -->

## Conclusion
<!-- Agent, after the run: did the hypothesis hold? Mechanistic explanation of why it
     worked or failed — not just the numbers. -->