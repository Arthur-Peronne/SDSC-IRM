---
model_name: AE3dAsymResSeparableV2
summary: 'Champion replicate 5/N, no HP change: n=5 spread already matches dim=60''s
  noise floor, one more point for a rounder sample before wrapping the replicate series'
parent: 02343abb
id: de6fd0e4
status: completed
verdict: FAILURE
created_at: '2026-07-13T19:16:19+00:00'
metric:
  primary:
    name: avg_validation_R2_mean
    value: 0.792444
    direction: maximize
---

# Trial de6fd0e4 — AE3dAsymResSeparableV2 — FAILURE

<!-- ===== written BEFORE the run (agent) ===== -->

## Hypothesis
Champion replicate series n=5: mean 0.7796, std≈0.0260, already consistent with
dim=60's ~0.03-0.04 noise floor. One more replicate for a slightly rounder n=6
before wrapping this series and writing the final campaign conclusion — no new
hypothesis, pure variance-reduction continuation.

## Implementation
No change to `configs/autoencoder.yaml` — identical to the CHAMPION (02343abb):
lr=6e-4, weight_decay=0.0, dropout_rate=0.0, noise_std=0.0, patience=50,
latent_dimensions=8, seed=0 unchanged.

<!-- ===== written AFTER the run ===== -->

## Results
- **validation_R2_mean per run:** 0c91c956: 0.792444
- **avg_validation_R2_mean:** 0.792444
- **delta_vs_champion** (display only): -0.013166
- **validation_MSE_mean** (mean, non-decisional): 162.082520
- **MLflow Run IDs:** 0c91c9568d9747d7b9f6bae4458fcd71

## Training Dynamics
Best epoch 53, early stopping at 103 — very close to the champion's own 52/102.
Val loss (0.000615) mid-range across the replicate series. Clean, unremarkable.

## Conclusion
Champion replicate series complete at n=6: {0.8056, 0.7655, 0.7991, 0.7856,
0.7420, 0.7924}, mean=0.7817, sample std≈0.0238 — a stable, well-characterized
estimate now, consistent with dim=60's noise floor. Wrapping this series here
(n=6, matching dim=60's own restraint about not over-investing purely in
variance work). With ~21 trials of budget remaining, next step: build matched-n
replicate samples for the other near-tied single-run candidates (baseline
noise_std=0.0001 currently n=2, lr=7e-4 currently n=1, patience=45 currently n=1)
so they can be fairly compared against this champion mean rather than against a
single lucky/unlucky draw. This turns the remaining budget into a genuine
small-scale comparison across the plausible-best configs instead of more
single-shot probing.