---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2_strided
summary: Replace MaxPool3d with strided Conv3d for all downsampling (pool1, z_pool3, block-level pooling), testing learnable downsampling vs fixed pooling
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
All HPs are now well-calibrated (lr=8e-4, dropout=0.3, weight_decay=1e-5, patience=20). The campaign 2 summary identified "replacing MaxPool3d with strided conv" as an untested architectural direction. The current champion uses MaxPool3d (fixed, non-learnable) for all downsampling: pool1, z_pool3, and block-level pooling. Replacing with strided Conv3d makes downsampling learnable — the network can adapt its pooling strategy to the data rather than using a fixed operation. This could capture more relevant spatial features for the classification task. The strided conv has the same receptive field as MaxPool3d (2×2) but learns what to preserve. I predict classification_accuracy_val > 0.6250.

## Implementation
In `src/models/ae_models.py` (mutable file), created new class `AutoEncoder3D_AsymResSeparableV2_strided`:
- Replaced `nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))` with `nn.Conv3d(8, 8, kernel_size=(1,2,2), stride=(1,2,2))` in pool1
- Replaced `nn.MaxPool3d(kernel_size=(2,1,1), stride=(2,1,1))` with `nn.Conv3d(32, 32, kernel_size=(2,1,1), stride=(2,1,1))` in z_pool3
- Replaced `nn.MaxPool3d(kernel_size=2, stride=2)` with `nn.Conv3d(out_channels, out_channels, kernel_size=2, stride=2)` in Conv3DBlock and SeparableConv3DBlock (block-level pooling)
- All other architecture identical to champion a2a3d9d1
- Config unchanged: `lr=8e-4`, `dropout_rate=0.3`, `weight_decay=1e-5`, `patience=20`, `latent_dimensions=20`

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