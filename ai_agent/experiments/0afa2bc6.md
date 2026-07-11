---
model_name: AE3dAsymResSeparableV2
summary: Test dropout=0.06 at lr=6e-4 — completing the dropout-transfer picture at
  a third lr point
parent: 3e07b08d
id: 0afa2bc6
status: completed
verdict: FAILURE
created_at: '2026-07-11T07:36:21+00:00'
metric:
  primary:
    name: avg_validation_R2_mean
    value: 0.784511
    direction: maximize
---

# Trial 0afa2bc6 — AE3dAsymResSeparableV2 — FAILURE

<!-- ===== written BEFORE the run (agent) ===== -->

## Hypothesis
`dropout=0.06` clearly hurt at `lr=8e-4` (delta -0.033, non-noise) but was absorbed
without instability at `lr=5e-4` (delta -0.036, but stable, trial 3087e561). `lr=6e-4`
is a third confirmed-good, stable point (trial 1f00c6a2, 0.802). Testing
`dropout=0.06` there completes the picture: does the dropout ceiling found at 8e-4
transfer to lr values close to it (6e-4) or only fail specifically at 8e-4 itself?

## Implementation
Two-field change relative to the champion: `lr: 8e-4 -> 6e-4` AND
`dropout_rate: 0.05 -> 0.06`. `weight_decay`, `noise_std`, `patience` unchanged from
the baseline defaults. No architectural change.

<!-- ===== written AFTER the run ===== -->

## Results
- **validation_R2_mean per run:** b60df0db: 0.784511
- **avg_validation_R2_mean:** 0.784511
- **delta_vs_champion** (display only): -0.043213
- **validation_MSE_mean** (mean, non-decisional): 163.735809
- **MLflow Run IDs:** b60df0db07e44f9bab202b8a55a1f59f

## Training Dynamics
<!-- Agent, after the run: stability, convergence speed, spikes, plateau, early stopping. -->

## Conclusion
<!-- Agent, after the run: did the hypothesis hold? Mechanistic explanation of why it
     worked or failed — not just the numbers. -->