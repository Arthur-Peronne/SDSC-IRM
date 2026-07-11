---
model_name: AE3dAsymResSeparableV2
summary: Stack weight_decay=1e-6 on lr=5e-4 + dropout=0.05 — the "other" good lr point,
  never combined with a second regularizer
parent: 3e07b08d
id: d132fbfc
status: completed
verdict: FAILURE
created_at: '2026-07-11T06:39:48+00:00'
metric:
  primary:
    name: avg_validation_R2_mean
    value: 0.802188
    direction: maximize
---

# Trial d132fbfc — AE3dAsymResSeparableV2 — FAILURE

<!-- ===== written BEFORE the run (agent) ===== -->

## Hypothesis
`weight_decay=1e-6` stacked on the champion's `lr=8e-4`+`dropout=0.05` was neutral
(trial de0f5947). All weight_decay/noise_std stacking so far has only been tried at
lr=8e-4. Trial 11530c52 established `lr=5e-4`+`dropout=0.05` as a second good,
stable point (0.808). This tests whether a tiny weight_decay addition behaves the
same way (neutral) at this different lr, completing the regularizer-stacking
picture across both known-good lr values rather than only the champion's.

## Implementation
Two-field change relative to the champion: `lr: 8e-4 -> 5e-4` AND
`weight_decay: 0.0 -> 1e-6`. `dropout_rate=0.05` unchanged. `noise_std`, `patience`
unchanged from the baseline defaults. No architectural change.

<!-- ===== written AFTER the run ===== -->

## Results
- **validation_R2_mean per run:** a6dd2137: 0.802188
- **avg_validation_R2_mean:** 0.802188
- **delta_vs_champion** (display only): -0.025536
- **validation_MSE_mean** (mean, non-decisional): 149.175751
- **MLflow Run IDs:** a6dd21373f494741a42dfa0459d7d87c

## Training Dynamics
<!-- Agent, after the run: stability, convergence speed, spikes, plateau, early stopping. -->

## Conclusion
<!-- Agent, after the run: did the hypothesis hold? Mechanistic explanation of why it
     worked or failed — not just the numbers. -->