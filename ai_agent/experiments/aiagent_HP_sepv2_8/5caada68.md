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
Best epoch 54, early stopping at 104. Val loss (0.000671) on the higher side,
per-patient std elevated (0.1226) though test std (0.0800) is normal — one
partial outlier patient likely, not a systemic instability like the dropout=0.01
or lr=8e-4 collapses.

## Conclusion
Baseline matched sample COMPLETE at n=6: {0.797717, 0.744314, 0.781888, 0.798491,
0.793813, 0.759396}, mean=0.7793, std≈0.0226. Compared to the champion's n=6
(mean=0.7817, std≈0.0238): a difference of only 0.0024, a small fraction of
either sample's own std — this is about as clean a statistical tie as this noise
level allows. **The noise_std=0 vs 0.0001 question is now definitively closed:
no real difference at dim=8**, fully consistent with (and much better supported
than) the earlier n=3-vs-n=2 comparison. Keeping noise_std=0 in the final
recommended config remains reasonable (one fewer nonzero knob, Occam's razor) but
should be reported as a tie, not an improvement. Moving to the next matched-n
comparison: lr=7e-4 (currently n=1) up to n=6.