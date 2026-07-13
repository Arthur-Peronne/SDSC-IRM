---
model_name: AE3dAsymResSeparableV2
summary: lr 6e-4 -> 3e-4, all else unchanged (checking whether the dim=60 lr-optimum
  shifts at dim=8)
parent: 0d2e0fa2
id: da8aad93
status: completed
verdict: FAILURE
created_at: '2026-07-13T13:59:51+00:00'
metric:
  primary:
    name: avg_validation_R2_mean
    value: 0.772862
    direction: maximize
---

# Trial da8aad93 — AE3dAsymResSeparableV2 — FAILURE

<!-- ===== written BEFORE the run (agent) ===== -->

## Hypothesis
At dim=60, lr showed a clean monotonic-then-cliff curve: 1e-5 -> 6e-4 improving
(diminishing returns), then a clean failure at 7e-4/8e-4 (§1 of the dim=60 report).
That peak was found on a ~60-dim bottleneck; dim=8 is a much smaller model with
far fewer effective degrees of freedom near the bottleneck, so the loss landscape
there could be sharper/more sensitive to step size. I test one point below the
transferred baseline (3e-4, which was itself CHAMPION at dim=60, R²=0.7874, just
short of the 6e-4 peak) to see whether the dim=8 optimum sits lower than 6e-4. If
3e-4 beats the 0.7977 baseline, the lr-peak shifts down with capacity and I'll
refine further below it; if it's flat or worse, 6e-4 stands confirmed at dim=8 too
and I'll stop probing this axis and move to noise_std/dropout instead.

## Implementation
`configs/autoencoder.yaml`: `lr: 6e-4 -> 3e-4`. Everything else identical to the
BASELINE (0d2e0fa2): weight_decay=0.0, dropout_rate=0.0, noise_std=0.0001,
patience=50, latent_dimensions=8.

<!-- ===== written AFTER the run ===== -->

## Results
- **validation_R2_mean per run:** b181a5c5: 0.772862
- **avg_validation_R2_mean:** 0.772862
- **delta_vs_champion** (display only): -0.024855
- **validation_MSE_mean** (mean, non-decisional): 177.211395
- **MLflow Run IDs:** b181a5c58c6142e693ac167eb9121189

## Training Dynamics
<!-- Agent, after the run: stability, convergence speed, spikes, plateau, early stopping. -->

## Conclusion
<!-- Agent, after the run: did the hypothesis hold? Mechanistic explanation of why it
     worked or failed — not just the numbers. -->