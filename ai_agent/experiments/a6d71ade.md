---
model_name: AE3dAsymResSeparableV2
summary: Interaction re-check — retest weight_decay=1e-6 (near-neutral under patience=30)
  now under the champion's patience=60, completing the regularizer-under-new-schedule
  sweep
parent: bed745a0
id: a6d71ade
status: completed
verdict: FAILURE
created_at: '2026-07-13T01:06:55+00:00'
metric:
  primary:
    name: avg_validation_R2_mean
    value: 0.810815
    direction: maximize
---

# Trial a6d71ade — AE3dAsymResSeparableV2 — FAILURE

<!-- ===== written BEFORE the run (agent) ===== -->

## Hypothesis
weight_decay=1e-6 was the mildest FAILURE found (-0.0024, within run-to-run noise)
under patience=30, and dropout's re-test just showed that a longer schedule makes a
weight-space capacity penalty MORE costly, not less (dropout's failure widened from
-0.0151 to -0.0410). weight_decay is a much gentler penalty than dropout, but the
same directional logic should apply: I predict another FAILURE, plausibly larger in
magnitude than the original -0.0024 (more epochs for the same per-epoch penalty to
compound), completing the picture that this architecture's capacity should not be
taxed at all at latent_dim=60, regardless of schedule length or regularizer type.

## Implementation
Single-field change in configs/autoencoder.yaml: weight_decay 0.0 -> 1e-6, on top of
the champion's lr=6e-4, noise_std=0.0001, patience=60. dropout_rate=0 unchanged. No
architecture change.

<!-- ===== written AFTER the run ===== -->

## Results
- **validation_R2_mean per run:** 09db3811: 0.810815
- **avg_validation_R2_mean:** 0.810815
- **delta_vs_champion** (display only): -0.004017
- **validation_MSE_mean** (mean, non-decisional): 142.384888
- **MLflow Run IDs:** 09db381174c041afa925483db972ef29

## Training Dynamics
Best epoch 65 (val 0.000540), close to the champion's epoch 92/0.000526 — a small
gap, not the dramatic degradation seen with dropout. Early stopped at 125 (vs 152),
a somewhat shorter run than the champion, but the overall shape (lr timing at
patience_scheduler=12, noise pattern) tracks closely throughout.

## Conclusion
Hypothesis confirmed but the magnitude prediction was off: FAILURE at -0.0040, close
to (not clearly larger than) the original -0.0024 under patience=30 — weight_decay's
near-neutral character held even under a schedule more than 2x longer, unlike
dropout's failure which roughly tripled. This is consistent with the two
regularizers' different mechanisms: dropout's stochastic capacity cut compounds with
every additional epoch it is active, while weight_decay's smooth, deterministic
shrinkage does not accumulate damage the same way — a longer schedule doesn't
meaningfully change how much it costs. This completes the regularizer-under-new-schedule
sweep: dropout and weight_decay both remain rejected at any schedule length tested,
while noise_std remains a confirmed contributor. Champion remains bed745a0. The
5-axis sweep and its main pairwise interactions with the found optimum are now
exhaustively mapped (18 trials); remaining budget is better spent on control
replicates to quantify batch_size=1 run-to-run variance around the champion, which
would put a confidence band on every delta reported so far.
