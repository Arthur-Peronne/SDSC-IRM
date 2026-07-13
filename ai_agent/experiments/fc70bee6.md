---
model_name: AE3dAsymResSeparableV2
summary: 'Baseline replicate 3/6 (noise_std=0.0001): building a matched n=6 sample
  against the champion''s n=6 for a fair mean-vs-mean comparison'
parent: 0d2e0fa2
id: fc70bee6
status: completed
verdict: FAILURE
created_at: '2026-07-13T19:35:10+00:00'
metric:
  primary:
    name: avg_validation_R2_mean
    value: 0.781888
    direction: maximize
---

# Trial fc70bee6 — AE3dAsymResSeparableV2 — FAILURE

<!-- ===== written BEFORE the run (agent) ===== -->

## Hypothesis
The champion (noise_std=0) now has a solid n=6 estimate (mean 0.7817, std≈0.0238).
The baseline (noise_std=0.0001) only has n=2 (mean 0.7710) — too few to compare
fairly. Rather than declare noise_std's effect settled on an unequal footing, I am
building the baseline up to a matched n=6 (this is replicate 3 of 4 needed:
trials 19-22) so the two means can be compared on equal terms. No new hypothesis
on the HP itself — this is about sample-size fairness for the campaign's final
conclusion.

## Implementation
`configs/autoencoder.yaml`: `noise_std: 0.0 -> 0.0001` (baseline's value),
everything else unchanged: lr=6e-4, weight_decay=0.0, dropout_rate=0.0,
patience=50, latent_dimensions=8, seed=0.

<!-- ===== written AFTER the run ===== -->

## Results
- **validation_R2_mean per run:** 0c1b5365: 0.781888
- **avg_validation_R2_mean:** 0.781888
- **delta_vs_champion** (display only): -0.023722
- **validation_MSE_mean** (mean, non-decisional): 168.705353
- **MLflow Run IDs:** 0c1b5365013544bd97315c799983675b

## Training Dynamics
<!-- Agent, after the run: stability, convergence speed, spikes, plateau, early stopping. -->

## Conclusion
<!-- Agent, after the run: did the hypothesis hold? Mechanistic explanation of why it
     worked or failed — not just the numbers. -->