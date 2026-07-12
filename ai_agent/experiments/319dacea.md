---
model_name: AE3dAsymResSeparableV2
summary: Test small input-space denoising (noise_std 0.0 -> 0.0001) on top of the
  champion's lr=6e-4, the one regularization mechanism not yet tried (dropout and
  weight_decay both failed)
parent: 97d513a3
id: 319dacea
status: completed
verdict: CHAMPION
created_at: '2026-07-12T20:44:49+00:00'
metric:
  primary:
    name: avg_validation_R2_mean
    value: 0.79996
    direction: maximize
---

# Trial 319dacea — AE3dAsymResSeparableV2 — CHAMPION

<!-- ===== written BEFORE the run (agent) ===== -->

## Hypothesis
Both dropout (0.05, weight-space stochastic) and weight_decay (1e-6, weight-space
smooth) failed on top of the champion, the former badly, the latter negligibly —
consistent with the champion's already-tight train/val gap (0.045) leaving little
room for regularization to help. noise_std is mechanistically different: it perturbs
the INPUT rather than the weights/activations, forcing the encoder to learn features
robust to small voxel-intensity noise rather than penalizing capacity directly. I
test a small value (0.0001) — deliberately far below the AEoptuna reference of 0.002,
since a similar denoising magnitude caused a catastrophic failure (R2 -0.185) in the
dim=240 campaign at its (much lower) baseline lr. Given the pattern so far, I predict
another FAILURE (the tight gap argument applies here too), but a distinct mechanism
is still worth the one test to complete the regularization sweep before moving to
patience.

## Implementation
Single-field change in configs/autoencoder.yaml: noise_std 0.0 -> 0.0001, on top of
the champion's lr=6e-4. weight_decay=0, dropout_rate=0 (both reverted), patience=30
unchanged. No architecture change.

<!-- ===== written AFTER the run ===== -->

## Results
- **validation_R2_mean per run:** c8052396: 0.799960
- **avg_validation_R2_mean:** 0.799960
- **delta_vs_champion** (display only): +0.002720
- **validation_MSE_mean** (mean, non-decisional): 150.229004
- **MLflow Run IDs:** c80523969edd4e969b6dfbe0d634bb60

## Training Dynamics
Similar raw-lr noise as the champion (epoch 7: val 0.002473 vs epoch 6's 0.001076;
epoch 28: 0.001356) — the small input noise did not visibly add to or reduce that.
Best epoch 45 (val loss 0.000570, the lowest of the campaign), early stopped at 75 —
essentially the same shape and length as the champion (best 48, stop 78), just
converging to a marginally better point.

## Conclusion
Hypothesis partially confirmed — I predicted another FAILURE but got the third
CHAMPION of the campaign: avg_validation_R2_mean 0.7972 -> 0.8000 (+0.0027), a small
but genuine reversal of the "tight gap leaves no room" pattern from the two prior
regularization attempts. Train R2 (0.8404) is essentially flat vs the champion's
(0.8422) — unlike dropout, this mechanism did not cost train-set fit — while val R2
improved slightly. This is consistent with the mechanistic distinction I expected:
input-space noise acts like a light data-augmentation, encouraging robustness to
voxel-level noise the scanner/registration pipeline already introduces, rather than
taxing model capacity the way dropout or weight_decay do. The gain is small enough to
be partly within run-to-run noise (as weight_decay's -0.0024 likely was), but its
sign is the opposite of the other two regularizers', which is itself informative.
Given a small positive result at 0.0001, the natural next test is a modest increase
(e.g. 0.0003) to see whether this specific mechanism has more headroom, while staying
well below the 0.002 magnitude that caused catastrophic failure in the dim=240
campaign.
