---
model_name: AE3dAsymResSeparableV2
summary: 'Control replicate #3 — exact champion config again, given the surprisingly
  wide spread (0.8148/0.8050/0.7396) seen in the first two replicates, to firm up
  the variance estimate with a 4-point sample'
parent: bed745a0
id: bc1bf3de
status: completed
verdict: FAILURE
created_at: '2026-07-13T02:07:56+00:00'
metric:
  primary:
    name: avg_validation_R2_mean
    value: 0.778275
    direction: maximize
---

# Trial bc1bf3de — AE3dAsymResSeparableV2 — FAILURE

<!-- ===== written BEFORE the run (agent) ===== -->

## Hypothesis
The 3-point sample so far (0.8148, 0.8050, 0.7396) has an unexpectedly wide spread
(0.075) for an identical, fixed-seed config — enough to make most of the campaign's
fine-grained (sub-0.03) verdicts statistically ambiguous. A 4th point will help
distinguish two possibilities: (a) the true distribution is genuinely this wide
(batch_size=1 stochastic training is highly sensitive to non-deterministic GPU op
ordering at this architecture/data scale), in which case a 4th draw should plausibly
land anywhere in or beyond the 0.74-0.81 range already seen, or (b) 2656977c's 0.7396
was itself a rare low outlier (e.g. an unlucky early basin) and most draws cluster
nearer 0.80-0.81, in which case this 4th point should land back in that tighter band.

## Implementation
No change to configs/autoencoder.yaml — exact champion config (lr=6e-4,
weight_decay=0.0, dropout_rate=0.0, noise_std=0.0001, patience=60). No architecture
change.

<!-- ===== written AFTER the run ===== -->

## Results
- **validation_R2_mean per run:** 3b275ecf: 0.778275
- **avg_validation_R2_mean:** 0.778275
- **delta_vs_champion** (display only): -0.036557
- **validation_MSE_mean** (mean, non-decisional): 168.936478
- **MLflow Run IDs:** 3b275ecfd16541489e33ee6a751da99a

## Training Dynamics
Best epoch 69 (val 0.000641) — between the original champion's 0.000526 and the
worst replicate's 0.000739, landing this 4th draw roughly in the middle of the
observed range. Train R2 (0.8637) is close to the champion's (0.8948), but val R2
(0.7783) trails well behind — the train/val gap here (0.086) is the widest of the
4 replicates, suggesting this run's model fit training data about as well as usual
but generalized somewhat worse.

## Conclusion
4-point sample of the identical config: {0.8148, 0.8050, 0.7396, 0.7783}. Mean≈0.784,
range 0.075, roughly 3 of 4 draws below the originally-recorded champion value —
confirming possibility (a) from the hypothesis: the variance is genuinely this wide,
not driven by one outlier, and 0.8148 looks like it sits on the upper side of the
distribution rather than at its center. This is a materially important campaign-level
finding: at batch_size=1 with this architecture/data, single-run avg_validation_R2_mean
has a standard deviation on the order of 0.03, meaning roughly two-thirds of the HP
comparisons made from trial 8 onward (deltas mostly in the ±0.01 to ±0.02 range) are
not statistically resolved by a single run each. The campaign's verdict rule
(deterministic, single-run comparison per experiment.yaml) is a reasonable and
necessary simplification for tractability, but its practical implication is that
"champion" here should be read as "best point observed," not "provably best
in expectation." Given 4 points already characterize the noise floor reasonably well
and further replicates have diminishing marginal information, remaining budget is
better spent on real HP exploration — interpreting future deltas against this ~0.03
noise floor rather than treating any single-run improvement as conclusively real.
