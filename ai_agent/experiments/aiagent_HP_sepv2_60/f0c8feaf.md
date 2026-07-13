---
model_name: AE3dAsymResSeparableV2
summary: Fifth replicate of patience=45, bringing its sample to parity (n=5) with
  patience=60, for the final variance comparison in the campaign report
parent: bc589070
id: f0c8feaf
status: completed
verdict: FAILURE
created_at: '2026-07-13T08:23:29+00:00'
metric:
  primary:
    name: avg_validation_R2_mean
    value: 0.813083
    direction: maximize
---

# Trial f0c8feaf — AE3dAsymResSeparableV2 — FAILURE

<!-- ===== written BEFORE the run (agent) ===== -->

## Hypothesis
patience=45 has 4 points (0.8089, 0.8060, 0.7913, 0.7954; std≈0.0073); patience=60
has 5 (std≈0.0346). A 5th patience=45 draw puts both configs on equal sample size for
the final report's headline variance comparison. Given the tempering seen with
patience=49's 3rd point, I hold this prediction loosely: the tight cluster could
still widen, but even a moderate widening would likely leave patience=45
meaningfully tighter than patience=60/50's ~0.035 spread.

## Implementation
No change to configs/autoencoder.yaml — same config as 69832c74/59547f27/3fbb7661/
10c18786 (lr=6e-4, noise_std=0.0001, patience=45). No architecture change. Fifth
replicate of the patience=45 config.

<!-- ===== written AFTER the run ===== -->

## Results
- **validation_R2_mean per run:** 507b5697: 0.813083
- **avg_validation_R2_mean:** 0.813083
- **delta_vs_champion** (display only): -0.001756
- **validation_MSE_mean** (mean, non-decisional): 140.997147
- **MLflow Run IDs:** 507b56973ab74a83b6c1db708755e4fd

## Training Dynamics
Best epoch 51 (val 0.000535), the earliest-best and highest-val-loss-lowest draw of
all 5 patience=45 replicates, yet the closest to the champion's own value
(-0.0018) — a short, efficient run rather than a long noisy one.

## Conclusion
5th point for patience=45: {0.8089, 0.8060, 0.7913, 0.7954, 0.8131}, mean≈0.8029,
std≈0.0083 — the tight cluster held up at n=5, now on equal footing with
patience=60's n=5 (std≈0.0346), a >4x difference in spread that has survived every
additional replicate. This closes the campaign's schedule-length variance
investigation with a well-supported (though not definitively proven) finding:
patience values landing on patience_scheduler=9 (45, 49) show both a slightly higher
mean (~0.80) and markedly lower run-to-run variance than patience_scheduler=10-12
(50, 60, mean~0.79, std~0.036).

Per explicit user direction, this closes the patience axis for the remainder of the
campaign: no further patience changes will be made. Remaining trials will focus on
combinations among lr, weight_decay, dropout_rate, and noise_std — an axis-interaction
gap correctly flagged as under-explored relative to the extensive patience/schedule
investigation above. Champion remains bc589070 (patience=50, lr=6e-4, noise_std=0.0001).
