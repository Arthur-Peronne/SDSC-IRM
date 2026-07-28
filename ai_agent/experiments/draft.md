---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2
summary: Increase patience 20->40 on champion (lr=8e-4, dropout=0.3, weight_decay=1e-5), testing if higher early stopping patience allows fuller convergence
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
Patience is the ONLY hyperparameter never tested from its default value (20) in this campaign. The champion (`lr=8e-4`, `dropout=0.3`) converges relatively quickly (early stopping at epochs ~50-67). With `patience=20`, the network stops 20 epochs after the best validation metric. If the validation metric has natural noise (program.md notes ~0.03-0.4 std dev on R²), the early stopping might trigger too early, before the network has fully converged. Doubling patience to 40 would give the network more epochs to find a better minimum, while the early stopping mechanism still prevents overfitting. I predict classification_accuracy_val ≥ 0.6250, with potentially higher R² due to fuller convergence.

## Implementation
In `configs/autoencoder.yaml` (only mutable file), branching from champion a2a3d9d1:
- `patience: 20 -> 40` (early stopping patience doubled)
- Unchanged: `lr=8e-4`, `dropout_rate=0.3`, `weight_decay=1e-5`, `noise_std=0.0`, `model_name=AE3dAsymResSeparableV2`, `latent_dimensions=20`

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