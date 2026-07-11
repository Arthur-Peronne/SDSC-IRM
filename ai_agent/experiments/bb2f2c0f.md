---
model_name: AE3dAsymResSeparableV2
summary: Third control replicate of the champion (lr=8e-4, dropout=0.05), with remaining
  campaign budget dedicated to a robust confidence estimate
parent: 3e07b08d
id: bb2f2c0f
status: completed
verdict: FAILURE
created_at: '2026-07-11T04:01:46+00:00'
metric:
  primary:
    name: avg_validation_R2_mean
    value: 0.779857
    direction: maximize
---

# Trial bb2f2c0f — AE3dAsymResSeparableV2 — FAILURE

<!-- ===== written BEFORE the run (agent) ===== -->

## Hypothesis
With trials 04d1d849 through 1a7b6adf, the dropout/lr/weight_decay/noise_std/patience
combination space around the champion is now thoroughly mapped: no single-axis or
combined variation has exceeded 0.828, and two independent replicates of the exact
champion config landed at 0.803/0.805 (below even the old baseline). With 5 trials
of budget left and diminishing value in yet more new HP points (per trial 1a7b6adf's
conclusion), the most useful remaining use of the campaign is building a firmer
estimate of this configuration's typical performance for the final summary. This is
a third exact replicate, not a new idea — deliberate in the sense of directly
serving the campaign's final, honest conclusion about the champion's expected value.

## Implementation
No change to `configs/autoencoder.yaml` relative to the champion (`lr=8e-4`,
`dropout_rate=0.05`, all other fields at baseline defaults, `seed=0` untouched).

<!-- ===== written AFTER the run ===== -->

## Results
- **validation_R2_mean per run:** 70856698: 0.779857
- **avg_validation_R2_mean:** 0.779857
- **delta_vs_champion** (display only): -0.047867
- **validation_MSE_mean** (mean, non-decisional): 162.334946
- **MLflow Run IDs:** 70856698927e4d02afc579eef136b89e

## Training Dynamics
Ran 86 epochs, early-stopped at `best_epoch=56` (+30 = 86). Best `val_loss` was
0.000616, the worst of the three exact-champion replicates so far. `validation_R2_std`
(0.101) is elevated relative to the champion's own 0.059, though not extreme.

## Conclusion
Third exact replicate of the champion config: {0.828, 0.803, 0.805, 0.780}. Mean
≈0.804, and the spread keeps widening rather than tightening around 0.828 — this
replicate is the lowest yet, below both previous replicates. This further confirms
(now with 3 independent samples) that 0.828 was a favorable outlier draw, not the
typical outcome of `lr=8e-4` + `dropout=0.05`; a realistic expectation for this
config is closer to 0.78-0.83, materially overlapping the old baseline's single
0.8075 sample. Champion remains 3e07b08d per the fixed ledger rule. Given the user
has now approved extending the campaign to `max_trials=50` (mid-campaign, applied as
a direct commit to experiment.yaml outside the trial mechanism, not through this
trial), the remaining ~1 replicate originally planned is enough for this data point;
subsequent trials will resume the deliberate-HP-change protocol now that the extra
budget is confirmed.