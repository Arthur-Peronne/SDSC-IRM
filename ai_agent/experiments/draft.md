---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2
summary: Fine-tune lr within the bracketed good band (5e-4 -> 6.5e-4), between the champion and the near-tying 8e-4
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
Four lr trials now bracket the productive region: 2e-4 (worse, under-explores),
5e-4=champion (0.8075), 8e-4 (tied, best raw val_loss), 1e-3 (worse, destabilizes).
The two known-good points (5e-4, 8e-4) straddle a region I have not directly sampled.
`lr=6.5e-4` tests the midpoint: if the relationship between lr and val performance is
smoothly unimodal with a peak between 5e-4 and 8e-4 (rather than 8e-4 itself being the
peak, or performance being flat across the whole band), this value has the best a
priori chance of beating the champion outright rather than merely tying it.

## Implementation
Single-field change in `configs/autoencoder.yaml`: `lr: 5e-4 -> 6.5e-4`. All other
hyperparameters unchanged from the baseline (weight_decay=0.0, dropout_rate=0.0,
noise_std=0.0, patience=30). No architectural change.

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