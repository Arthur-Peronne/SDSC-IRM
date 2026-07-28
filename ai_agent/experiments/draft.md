---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResidualV4
summary: Test AE3dAsymResidualV4 (pre-existing, highest historical R²=0.773806) with champion HPs (lr=8e-4, dropout=0.3, weight_decay=1e-5, patience=20)
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
The champion architecture (AE3dAsymResSeparableV2) is hard to beat — all architectural variations tested so far (strided conv, AE3dAsymResSeparable without V2) have failed. However, AE3dAsymResidualV4 has the highest historical R² (0.773806) among all pre-existing models, significantly above the champion's 0.7567. This architecture uses asymmetric residual connections with V4 modifications, which might produce better latent representations. Testing it with the fully validated champion HPs (lr=8e-4, dropout=0.3, weight_decay=1e-5, patience=20) is a fair comparison. I predict classification_accuracy_val > 0.6250.

## Implementation
In `configs/autoencoder.yaml` (only mutable file), branching from champion a2a3d9d1:
- `model_name: AE3dAsymResSeparableV2 -> AE3dAsymResidualV4`
- Unchanged: `lr=8e-4`, `dropout_rate=0.3`, `weight_decay=1e-5`, `patience=20`, `latent_dimensions=20`

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