---
model_name: AE3dAsymResSeparableV2SELateStrided
summary: Replace champion's 2 explicit anisotropic MaxPools with learnable depthwise
  strided convs
parent: 761cab78
id: 2dd4cd5f
status: completed
verdict: FAILURE
created_at: '2026-07-28T15:35:38+00:00'
metric:
  primary:
    name: classification_accuracy_val
    value: 0.616667
    direction: maximize
---

# Trial 2dd4cd5f — AE3dAsymResSeparableV2SELateStrided — FAILURE

<!-- ===== written BEFORE the run (agent) ===== -->

## Hypothesis
New architectural family (not another SE-placement variant — that direction is now closed after 3
trials: a581f44e, 761cab78, 09415e52). Champion `AE3dAsymResSeparableV2SELate`'s two explicit
anisotropic poolers (`pool1`: 32x128x128->32x64x64, `z_pool3`: 32x8x16x16->32x4x16x16) are
`nn.MaxPool3d`, a fixed, non-learnable rule that keeps only the single strongest activation per
window and discards the rest. I replace both with depthwise strided convolutions of the same
kernel/stride (channel count unchanged via `groups=channels`, so parameter cost stays negligible).
Mechanism: a learnable downsampling can combine values within each window instead of hard-selecting
one, potentially preserving fine structural gradients (e.g. wall-thickness variation) that a max
operation discards outright — detail that may carry group-discriminative signal even where it
doesn't change which single activation is the local max. Per-block internal downsampling (inside
`ResSeparableConv3DBlock`/`SeparableConv3DBlock`, shared with other architectures) is left as
MaxPool — only the two standalone ops unique to this model are touched, to keep the change
contained. SE placement (se3, se4) and all hyperparameters (`lr=1e-4`, `weight_decay=0`,
`dropout_rate=0`, `noise_std=0`, `patience=20`) kept identical to the champion.

## Implementation
New class `AutoEncoder3D_AsymResSeparableV2_SELateStrided` in `src/models/ae_models.py` (model_name
`AE3dAsymResSeparableV2SELateStrided`), copied from `AutoEncoder3D_AsymResSeparableV2_SELate`
(current champion) with exactly two lines changed: `pool1` becomes
`nn.Conv3d(8, 8, kernel_size=(1,2,2), stride=(1,2,2), groups=8)` and `z_pool3` becomes
`nn.Conv3d(32, 32, kernel_size=(2,1,1), stride=(2,1,1), groups=32)` — same kernel/stride/channel
count as the MaxPools they replace, now with learnable weights. No encoder-to-decoder path —
respects the no-skip-connections rule in `program.md`. Verified the new class builds, forward-passes,
and produces a (1,20) latent before launching the trial.

<!-- ===== written AFTER the run ===== -->

## Results
- **accuracy_test per run:** 0: 0.575000 | 1: 0.725000 | 2: 0.550000
- **classification_accuracy_val:** 0.616667
- **delta_vs_champion** (display only): -0.083333
- **validation_R2_mean** (mean, AE phase, non-decisional): 0.731634
- **AE MLflow Run IDs:** 9d572574d96249409f87fac3c92b669f b27b6b3dbf4e4cfc9dc32b6aa65242b4 2edd656d7a7b4ca88c27eb915ea0d870
- **Classification MLflow Run IDs:** c772dd8ef55442ee8ef7d4a9dd0c3337 c22b44442c8b47388bed80ced5775611 5f831208f93b4f288307ed0961bc43d3

## Training Dynamics
<!-- Agent, after the run: stability, convergence speed, spikes, plateau, early stopping. -->

## Conclusion
<!-- Agent, after the run: did the hypothesis hold? Mechanistic explanation of why it
     worked or failed — not just the numbers. -->