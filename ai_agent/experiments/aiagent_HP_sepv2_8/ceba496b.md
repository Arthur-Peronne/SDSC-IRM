---
model_name: AE3dAsymResSeparableV2
summary: 'Replicate of the BASELINE config (noise_std=0.0001), no HP change: fair
  n=2 mean-vs-mean comparison against the champion''s n=3 mean given the confirmed
  noise floor'
parent: 0d2e0fa2
id: ceba496b
status: completed
verdict: FAILURE
created_at: '2026-07-13T16:48:19+00:00'
metric:
  primary:
    name: avg_validation_R2_mean
    value: 0.744314
    direction: maximize
---

# Trial ceba496b — AE3dAsymResSeparableV2 — FAILURE

<!-- ===== written BEFORE the run (agent) ===== -->

## Hypothesis
The champion's 3-point spread (0.8056/0.7655/0.7991, mean 0.7901) now sits at or
below the single-run BASELINE value (0.797717) — the noise_std=0 "improvement"
looks like it may have been a favorable draw rather than a real effect. But
that's an n=3-vs-n=1 comparison, which is itself unfair. I replicate the BASELINE
config (noise_std=0.0001, everything else identical) once to get a same-footing
n=2 mean for the baseline before concluding the noise_std axis is genuinely flat
rather than mildly beneficial at dim=8.

## Implementation
`configs/autoencoder.yaml`: `noise_std: 0.0 -> 0.0001` (reverting to the
BASELINE's value), everything else unchanged: lr=6e-4, weight_decay=0.0,
dropout_rate=0.0, patience=50, latent_dimensions=8, seed=0.

<!-- ===== written AFTER the run ===== -->

## Results
- **validation_R2_mean per run:** b0a38f4b: 0.744314
- **avg_validation_R2_mean:** 0.744314
- **delta_vs_champion** (display only): -0.061296
- **validation_MSE_mean** (mean, non-decisional): 185.732208
- **MLflow Run IDs:** b0a38f4b2adb4e4eb461e88fad58605d

## Training Dynamics
Best epoch 43, early stopping at 93 — earlier and shorter than the original
BASELINE run (76/126). Val loss at best epoch (0.000705) is the worst of any
noise_std=0.0001 run seen so far, and per-patient std (0.1174) is the highest yet
recorded outside the lr=8e-4 collapse (trial 6, std 0.156). This single replicate
lands unusually low even for this noise band.

## Conclusion
BASELINE now n=2: {0.797717, 0.744314}, mean=0.7710, sample std≈0.0378 — wider
spread than the champion's 3-point std (0.0215), though from fewer points so less
reliable itself. Comparing means: champion 0.7901 (n=3) vs baseline 0.7710 (n=2),
difference +0.019, but the standard error on that difference (~0.03, combining
both samples' variances) is *larger* than the difference itself. Honest
conclusion: still not statistically resolvable — the point estimate mildly favors
the champion (noise_std=0) but not with confidence, and chasing full significance
here would need many more replicates for a marginal, practically-unimportant
effect (mirroring dim=60's own reflection about over-investing in variance
work). I am closing the noise_std question here: **no confident preference
between 0 and 0.0001**, defaulting to 0 (the champion's value) for simplicity
(one fewer nonzero knob) rather than because it's proven better. Pivoting the
remaining budget toward the axis with the largest, most robust effect size found
so far — lr — to refine its optimum more precisely (finer grid between 6e-4,
confirmed good, and 8e-4, confirmed catastrophic) rather than continuing to
replicate a sub-noise-floor question.