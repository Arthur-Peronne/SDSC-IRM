---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2
summary: Increase dropout_rate 0.1->0.3 on champion (lr=8e-4, dropout=0.1), testing if stronger bottleneck dropout pairs better with higher learning rate
parent: b3a18c7b

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
Trial 12 (b3a18c7b) found that `lr=8e-4` + `dropout=0.1` equals the champion accuracy (0.6083), but trial 11 (4682d5a1) with `lr=8e-4` alone only reached 0.5917 — suggesting the higher LR benefits from the regularization that dropout provides. Meanwhile, trial 3 showed that `dropout=0.3` + `lr=5e-4` was the first real improvement over baseline. The question is whether `dropout=0.3` + `lr=8e-4` combines the best of both worlds: the higher learning rate's faster convergence with stronger bottleneck regularization. Mechanistically, dropout=0.3 forces more redundancy in the 2048-d latent vector, which should pair well with a higher LR that might otherwise overfit. I predict classification_accuracy_val > 0.6083.

## Implementation
In `configs/autoencoder.yaml` (only mutable file), branching from champion b3a18c7b:
- `dropout_rate: 0.1 -> 0.3` (bottleneck dropout only, applied on flattened 2048-d vector)
- Unchanged: `lr=8e-4`, `weight_decay=1e-5`, `noise_std=0.0`, `patience=20`, `model_name=AE3dAsymResSeparableV2`, `latent_dimensions=20`

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