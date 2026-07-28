---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparable
summary: Switch from AE3dAsymResSeparableV2 to AE3dAsymResSeparable (pre-existing architecture with higher historical R²=0.795874 vs 0.7567), keeping champion HPs
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
The champion architecture (AE3dAsymResSeparableV2) has R²=0.7567. The pre-existing AE3dAsymResSeparable (without V2) has a higher historical R² of 0.795874 (from previous campaigns). While R² is not the judge metric, a higher R² suggests the architecture learns better reconstructions, which could translate to better latent representations for classification. The V2 variant added residual connections and separable convolutions but may have over-engineered the architecture for this small dataset. The original AsymResSeparable might be simpler and generalize better. I test it with the fully validated champion HPs (lr=8e-4, dropout=0.3, weight_decay=1e-5, patience=20). I predict classification_accuracy_val > 0.6250.

## Implementation
In `configs/autoencoder.yaml` (only mutable file), branching from champion a2a3d9d1:
- `model_name: AE3dAsymResSeparableV2 -> AE3dAsymResSeparable`
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