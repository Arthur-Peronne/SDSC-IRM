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
Convergence is drastically faster than the BASELINE: val loss is already near its final
order of magnitude after epoch 1 (e.g. dim=8: 0.0016 vs BASELINE's 0.072), whereas
BASELINE was still 0.07-0.14 at epoch 1. All 3 dims reach their best val loss between
epoch 3-4, then plateau/mildly worsen for the remaining 1-2 epochs (dim=8 best@3,
dim=40 best@4 after a one-epoch wobble@2, dim=200 best@3) — a first sign of
overshoot/noise at this lr, but contained (no divergence, no NaN) within the 5-epoch
window. `patience=30` never engages (irrelevant at this horizon). Train and val track
closely at all dims (no overfitting signature yet, consistent with only 5 epochs).

## Conclusion
Hypothesis confirmed, and by a large margin: `avg_validation_R2_mean` went from -0.5428
(BASELINE, still descending at cutoff) to 0.5212 (delta +1.064), i.e. the model now
explains variance rather than performing worse than the mean baseline. This supports the
mechanism directly: under the frozen 5-epoch budget, the BASELINE's low lr simply had not
made enough progress, and a 10x larger lr let each run do in ~3 epochs what previously
needed far more. The mild post-peak wobble at epoch 4-5 (esp. dim=40) suggests 5e-4 may
be at or slightly past the locally-stable step size for later epochs — worth watching if
a future campaign relaxes `n_epochs`, but it is not a failure mode here. New CHAMPION.