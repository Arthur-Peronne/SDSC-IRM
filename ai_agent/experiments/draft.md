---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2
summary: Increase lr 8e-4->1e-3 on champion (dropout=0.3, weight_decay=1e-5), testing if higher learning rate continues the lr=5e-4→8e-4 improvement trend
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
There's a clear trend: `lr=5e-4` + `dropout=0.3` (trial 3) = 0.6083, `lr=8e-4` + `dropout=0.3` (champion a2a3d9d1) = 0.6250. The higher LR consistently improves accuracy when paired with dropout=0.3. Trial 14 (8cd4bf76) showed that weight_decay=1e-4 fails catastrophically with this combo, so I revert it to 1e-5. The question is whether the LR improvement trend continues at `lr=1e-3`. Mechanistically, a higher LR allows the network to explore the loss landscape more aggressively and escape shallow local minima. With dropout=0.3 providing strong bottleneck regularization, the network should be able to handle the higher LR without overfitting. I predict classification_accuracy_val > 0.6250.

## Implementation
In `configs/autoencoder.yaml` (only mutable file), branching from champion a2a3d9d1:
- `lr: 8e-4 -> 1e-3` (25% increase in learning rate)
- `weight_decay: 1e-4 -> 1e-5` (revert trial 14's failed change back to champion value)
- Unchanged: `dropout_rate=0.3`, `noise_std=0.0`, `patience=20`, `model_name=AE3dAsymResSeparableV2`, `latent_dimensions=20`

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