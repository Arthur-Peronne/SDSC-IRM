---
model_name: AE3dAsymResSeparableV2
summary: 'lr 5e-4 -> 1.5e-3 (3x): test whether pushing lr further still helps, or
  the epoch4-5 wobble seen at 5e-4 turns into instability'
parent: e3b6e5ff
id: acf6f37d
status: completed
verdict: FAILURE
created_at: '2026-07-10T18:59:55+00:00'
metric:
  primary:
    name: avg_validation_R2_mean
    value: 0.441396
    direction: maximize
---

# Trial acf6f37d — AE3dAsymResSeparableV2 — FAILURE

<!-- ===== written BEFORE the run (agent) ===== -->

## Hypothesis
The current champion (e3b6e5ff, lr=5e-4) converged in ~3 epochs then showed a mild
post-peak wobble in epochs 4-5 (val loss ticking back up, most visibly at dim=40:
0.0010 -> 0.0046 at epoch 5) — a classic early signature of the step size becoming too
large relative to the loss landscape's curvature near the optimum, even though it did
not cause outright divergence. This is the last trial of the campaign (`max_trials=3`),
so rather than a fresh direction I test the natural follow-up to trial 2's confirmed
lever: push `lr` further (5e-4 -> 1.5e-3, 3x) to see whether, under the same frozen
5-epoch budget, the faster-convergence benefit continues (if the wobble was mere noise
and convergence speed still dominates) or the increase now costs more than it gains
(if the wobble was an early warning of instability). Either outcome is informative for
future campaigns' choice of `lr`, even though this trial may not beat the champion.

## Implementation
`configs/autoencoder.yaml`: `lr: 5e-4` -> `lr: 1.5e-3`. No other field touched
(`weight_decay`, `dropout_rate`, `noise_std`, `patience` unchanged from the champion;
architecture, data, split, `n_epochs` all frozen/untouched).

<!-- ===== written AFTER the run ===== -->

## Results
- **validation_R2_mean per run:** 8: 0.533355 | 40: 0.418283 | 200: 0.372551
- **avg_validation_R2_mean:** 0.441396
- **delta_vs_champion** (display only): -0.079804
- **validation_MSE_mean** (mean, non-decisional): 329.336416
- **MLflow Run IDs:** eb9199ed4ea447c4a7b6566e22da7865 1c42791ae5ab4bb48f40c1a4f00b58b4 6d36ef0a3f0c4370b43c9c19e56bd221

## Training Dynamics
The wobble seen only mildly at epoch 4-5 in the champion (lr=5e-4) now appears from
epoch 2 onward at all 3 dims and is larger: e.g. dim=8 best@1 (0.0015) -> worse@2
(0.0020) -> new best@3 (0.0011) -> worse@4 -> flat@5; dim=40 similarly, but its epoch-5
val loss (0.0054) is now worse than the champion's already-noisy epoch-5 (0.0046);
dim=200 tracks the same up-down pattern. No divergence/NaN, but the oscillation is
clearly earlier and larger than at lr=5e-4, and it scales with latent_dimensions: the
higher-capacity runs (40, 200) lose more R² relative to the champion than dim=8 does.

## Conclusion
Hypothesis's second branch held: pushing lr further (5e-4 -> 1.5e-3) crossed from
"faster-convergence still dominates" into "step-size noise dominates" — avg R² dropped
from 0.5212 to 0.4414 (delta -0.0798), below both the CHAMPION bar and the CANDIDATE
margin (best per-run 0.5334 < champion 0.5212 + 0.03 = 0.5512). This locates the
sweet-spot boundary for this architecture/epoch-budget between 5e-4 (net positive,
mild noise) and 1.5e-3 (net negative, noise scaling with latent_dimensions — dim=200
lost the most). Actionable for a future campaign: search lr in roughly [1e-4, 7e-4]
rather than pushing toward 1e-3+, and consider that higher latent_dimensions may need
comparatively smaller lr than lower ones. As FAILURE, `configs/autoencoder.yaml` was
reverted by the driver to the champion's values (lr=5e-4) — confirmed on disk.
Campaign budget (`max_trials=3`) is now spent; champion remains e3b6e5ff.