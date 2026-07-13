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
<!-- Agent, after the run: stability, convergence speed, spikes, plateau, early stopping. -->

## Conclusion
<!-- Agent, after the run: did the hypothesis hold? Mechanistic explanation of why it
     worked or failed — not just the numbers. -->