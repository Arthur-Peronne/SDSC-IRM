---
model_name: AE3dAsymResSeparableV2
summary: Stack weight_decay=1e-6 AND noise_std=0.0001 together (both individually
  neutral) on the champion's lr=8e-4 + dropout=0.05
parent: 3e07b08d
id: 1a7b6adf
status: completed
verdict: FAILURE
created_at: '2026-07-11T03:40:18+00:00'
metric:
  primary:
    name: avg_validation_R2_mean
    value: 0.803182
    direction: maximize
---

# Trial 1a7b6adf — AE3dAsymResSeparableV2 — FAILURE

<!-- ===== written BEFORE the run (agent) ===== -->

## Hypothesis
Trials de0f5947 (`weight_decay=1e-6` alone on the champion) and 5033c15b
(`noise_std=0.0001` alone on the champion) were both individually neutral (deltas
-0.017 and -0.030, inside/at the edge of the noise band). Neither showed the
catastrophic effect their larger counterparts did (weight_decay=1e-5, noise_std=0.002),
so both are "safe" at this magnitude. This trial fuses them: stacking two mechanistically
different, individually-harmless regularizers (deterministic L2 + input-space
corruption) together, on top of the already-working dropout, to test whether
they compound (either positively, if they attack different aspects of the same
generalization gap, or negatively, if the "regularization budget" is a real
constraint and even harmless individual additions become harmful in combination).

## Implementation
Two-field addition relative to the champion: `weight_decay: 0.0 -> 1e-6` AND
`noise_std: 0.0 -> 0.0001`. `lr=8e-4` and `dropout_rate=0.05` unchanged. `patience`
unchanged from the baseline default. No architectural change.

<!-- ===== written AFTER the run ===== -->

## Results
- **validation_R2_mean per run:** f126bb35: 0.803182
- **avg_validation_R2_mean:** 0.803182
- **delta_vs_champion** (display only): -0.024542
- **validation_MSE_mean** (mean, non-decisional): 149.093445
- **MLflow Run IDs:** f126bb35b9994d2d83f3d5d3d7b7a3ad

## Training Dynamics
Ran 114 epochs, early-stopped at `best_epoch=84` (+30 = 114) — a long, healthy
training window comparable to the champion's 80. Best `val_loss` was 0.000566, close
to the champion's 0.000493 and squarely inside the range of the other neutral
stacking trials. `validation_R2_std` (0.074) is unremarkable.

## Conclusion
No synergy detected: stacking both individually-neutral regularizers together
(delta -0.025) lands right where each did alone (weight_decay alone: -0.017;
noise_std alone: -0.030) — not better, not measurably worse either. This confirms
the "regularization budget" reading from earlier trials: dropout=0.05 at lr=8e-4 is
doing essentially all the useful regularization this architecture can absorb at
`latent_dim=240`; additional regularizers, alone or combined, are consistently
redundant rather than harmful or additive. Champion remains 3e07b08d. This closes
out the combination search — every mechanistically distinct addition to the
champion's core (lr, dropout) pairing has now been tried, individually and in this
one combined form, with no result exceeding 0.828. Remaining trials will be used to
consolidate confidence in the current champion rather than search further variations
in this well-mapped neighborhood.