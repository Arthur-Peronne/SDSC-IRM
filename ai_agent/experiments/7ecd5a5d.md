---
model_name: AE3dAsymResSeparableV2
summary: Last lr step (6e-4 -> 8e-4), matching the dim=240 champion exactly, to close
  the lr axis given sharply diminishing gains and the first visible raw-lr noise at
  6e-4
parent: 97d513a3
id: 7ecd5a5d
status: completed
verdict: FAILURE
created_at: '2026-07-12T19:28:56+00:00'
metric:
  primary:
    name: avg_validation_R2_mean
    value: 0.789637
    direction: maximize
---

# Trial 7ecd5a5d — AE3dAsymResSeparableV2 — FAILURE

<!-- ===== written BEFORE the run (agent) ===== -->

## Hypothesis
The last three lr increases won with sharply diminishing returns (+0.096, +0.040,
+0.010) and 6e-4 showed the first visible raw-lr noise (val loss jumping more within
an epoch window than at 3e-4), while still no divergence/NaN. This is the profile of
approaching a ceiling. I test lr=8e-4 (matching the dim=240 campaign's champion
exactly) as the deliberate last step on this axis: if the trend continues even
slightly, there is still headroom and I would consider one more step; if it plateaus,
regresses, or shows outright instability, that firmly closes the lr axis at 6e-4 (the
best seen so far) and moves exploration to weight_decay / dropout_rate / noise_std /
patience, none of which have been touched since the neutral baseline.

## Implementation
Single-field change in configs/autoencoder.yaml: lr 6e-4 -> 8e-4. All other opened
HPs unchanged (weight_decay=0, dropout_rate=0, noise_std=0, patience=30). No
architecture change.

<!-- ===== written AFTER the run ===== -->

## Results
- **validation_R2_mean per run:** 1c69fc89: 0.789637
- **avg_validation_R2_mean:** 0.789637
- **delta_vs_champion** (display only): -0.007603
- **validation_MSE_mean** (mean, non-decisional): 157.305435
- **MLflow Run IDs:** 1c69fc899a4a41f490c095e0cc9d8ff9

## Training Dynamics
Clearest instability yet: epoch 7 spikes to val=0.004516 (vs 0.000932 the epoch
before) — a 4-5x jump, the largest single-epoch excursion seen in this campaign,
before recovering. Several other rough patches follow (epoch 39: 0.001345, epoch 41:
0.001157) that were not present at 6e-4's already-noisier-than-3e-4 profile. Needed
3 scheduler decays (8e-4->4e-4->2e-4->1e-4->...) and 101 epochs (vs 78 at 6e-4) to
reach its best (epoch 71), i.e. slower AND rougher than the champion, not faster.

## Conclusion
Hypothesis resolved cleanly: FAILURE, avg_validation_R2_mean 0.7972 -> 0.7896
(-0.0076), and validation R2 std reversed its 4-trial improving streak (0.0869, up
from 0.0653) — both the mean and the uniformity got worse. This closes the lr axis:
6e-4 (champion 97d513a3) is the practical optimum for this architecture at
latent_dim=60 with batch_size=1, not 8e-4 despite that being what worked best at
latent_dim=240 — HPs did not fully transfer across bottleneck sizes, confirming the
campaign's opening hypothesis that dim=60 needed its own search rather than
inheriting dim=240's champion. Mechanistically, batch_size=1 makes each step a
single-sample gradient estimate; at 8e-4 the step size is large enough that some
unlucky samples produce a destabilizing update (the epoch-7 spike) even though the
scheduler/patience mechanism still recovers a usable model eventually — just a worse
and slower one. Exploration now moves to the other 4 opened HPs. The champion
(97d513a3, lr=6e-4) already has a fairly tight train/val gap (0.8422 vs 0.7972,
0.045) — much tighter than the neutral baseline's (0.183) — so classic
regularizers (dropout, weight_decay, noise) have less overfitting left to correct
than they would have against the baseline; testing them is still worthwhile but
expectations should be modest, and it is plausible some or all will FAIL by removing
capacity the model needs rather than curbing overfitting it no longer has.
