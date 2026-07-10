---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2
summary: Add mild L2 regularization (weight_decay 0.0 -> 1e-5), a deterministic alternative to dropout after trial 6a8a38d6's FAILURE
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
Trial 6a8a38d6 showed that `dropout_rate=0.15` reduces training fit but hurts
validation R2 (0.807 -> 0.799), likely because dropout's stochastic noise interacts
badly with `batch_size=1` (no batch averaging to smooth the per-step estimate). I set
`weight_decay: 0.0 -> 1e-5` instead: L2 regularization shrinks weight magnitudes
deterministically at every step, without adding forward-pass noise, so it should
target the same mild train/val gap (train_R2 0.859 vs val_R2 0.807 in the baseline)
without dropout's noise side-effect. I predict `avg_validation_R2_mean` improves
above 0.8075, or at worst degrades much less than trial 6a8a38d6 did.

## Implementation
Single-field change in `configs/autoencoder.yaml`: `weight_decay: 0.0 -> 1e-5`. All
other hyperparameters unchanged from the baseline (lr=5e-4, dropout_rate=0.0,
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