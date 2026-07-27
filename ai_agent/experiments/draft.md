---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2
summary: Increase weight_decay 1e-5->1e-4 (10x) on champion (lr=8e-4, dropout=0.3), testing mild L2 regularization that was never tried (trial 7 tested 100x which failed catastrophically)
parent: a2a3d9d1

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
Trial 7 (e4e99f58) tested `weight_decay=1e-3` (100x increase) on the old champion and everything collapsed: accuracy dropped to 0.2167 and R² to 0.4771 — classic over-regularization/underfitting. However, the campaign summary noted that a 10x step (1e-5→1e-4) was never tested and might behave completely differently. With the current champion's `dropout=0.3` already providing activation-level regularization, adding mild L2 regularization (weight_decay=1e-4) could complement it: dropout randomly zeros activations while L2 shrinks all weights, targeting different aspects of overfitting. The 10x step should be strong enough to matter but gentle enough not to hurt optimization. I predict classification_accuracy_val > 0.6250.

## Implementation
In `configs/autoencoder.yaml` (only mutable file), branching from champion a2a3d9d1:
- `weight_decay: 1e-5 -> 1e-4` (10x increase, L2 regularization on all weights)
- Unchanged: `lr=8e-4`, `dropout_rate=0.3`, `noise_std=0.0`, `patience=20`, `model_name=AE3dAsymResSeparableV2`, `latent_dimensions=20`

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