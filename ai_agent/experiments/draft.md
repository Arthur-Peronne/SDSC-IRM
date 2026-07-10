---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2
summary: "Fuse two prior leads: lr=8e-4 (trial f23a08a7, near-tie) with a much smaller dropout=0.05 (vs trial 6a8a38d6's failed 0.15)"
parent: 185cf97f

# ---- driver-written (leave null; the driver overwrites at lock/result) ----
id: null              # short sha of commit 1 == the frozen input == this trial's identity
status: draft         # draft -> completed | failed          (lifecycle, lowercase)
verdict: null         # BASELINE | CHAMPION | CANDIDATE | FAILURE   (judgement, UPPERCASE)
created_at: null
metric:
  primary: {name: avg_validation_R2_mean, value: null, direction: maximize}
---

# Trial <id> — <model_name> — <verdict>

<!-- ===== written BEFORE the run (agent) ===== -->

## Hypothesis
Fusing two prior leads rather than continuing single-axis search: (1) trial
f23a08a7 showed `lr=8e-4` converges faster (best_epoch 48 vs 66) to a tied/slightly
better raw val_loss than the champion, with no instability; (2) trial 6a8a38d6
showed `dropout=0.15` was too strong (delta -0.008, though within noise) but the
mechanism (regularizing against the baseline's train/val gap) is not necessarily
wrong at every magnitude — only 0.15 and weight_decay=1e-5 were tested, both
plausibly "too much". Faster convergence (from the higher lr) means fewer effective
epochs to overfit before early stopping, which could make a *small* dose of dropout
more additive than it was on the slower-converging baseline. I use `dropout=0.05`
(3x smaller than the failed 0.15) alongside `lr=8e-4`, predicting the combination
beats both the champion and the 8e-4-alone near-tie, rather than assuming either
change alone (already tested individually) will newly cross the line. Both f23a08a7
and 6a8a38d6 had their code reverted after FAILURE verdicts (per the driver's
contract), so the actual code base for this edit is the champion 185cf97f; `parent`
is set to 185cf97f to reflect that, with the fusion of ideas described here.

## Implementation
Two-field change in `configs/autoencoder.yaml`: `lr: 5e-4 -> 8e-4` AND
`dropout_rate: 0.0 -> 0.05`. `weight_decay`, `noise_std`, `patience` unchanged from
the baseline. No architectural change.

<!-- ===== written AFTER the run ===== -->

## Results
<!-- Filled automatically by the driver — leave empty. It writes, for a completed trial:
     per-run metric values (by repeat axis), the aggregated primary metric,
     delta_vs_champion (display only), the also_log means, and the MLflow run ids.
     For a mechanically failed trial it writes the failure reason instead. -->

## Training Dynamics
<!-- Agent, after the run: stability, convergence speed, spikes, plateau, early stopping. -->

## Conclusion
<!-- Agent, after the run: did the hypothesis hold? Mechanistic explanation of why it
     worked or failed — not just the numbers. -->