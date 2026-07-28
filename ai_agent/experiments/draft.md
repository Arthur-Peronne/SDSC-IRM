---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2SELateResBottleneck
summary: "Replace bottleneck_conv's plain conv stack with the existing ResConv3DBlock (same convs + same-stage residual add), extending the champion's own residual pattern to the bottleneck"
parent: 761cab78

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
Seventh distinct architectural mechanism this campaign (SE placement, CBAM, downsampling method, FC
depth, dilation, and bottleneck layer count all falsified/neutral; bottleneck normalization type was a
near-miss). The champion already uses residual blocks (`Res*` prefix) everywhere except
`bottleneck_conv`: `enc1`-`enc3` use `ResSeparableConv3DBlock`, every decoder stage uses
`ResUpSeparableConv3DBlock`, but `bottleneck_conv` (the two 3x3x3 convs right before the FC
projection) is plain, with no residual/shortcut path. I replace it with the already-existing
`ResConv3DBlock(64, 128, downsample=False)` — same two-conv/InstanceNorm/ReLU structure, plus a
same-stage identity-ish shortcut (1x1x1 conv since channels change 64->128) added before the final
ReLU. Mechanism: residual learning gives this stage an easier optimization path (can default toward
the shortcut projection if the two convs have little useful to add on top), the same reason it
presumably helps everywhere else in this architecture — testing whether that benefit extends to the
one place it was never applied. This is a genuinely different mechanism from the capacity-reduction
(`SimpleBottleneck`, failed) and normalization-swap (`BottleneckGN`, near-miss) ideas already tried at
this same location: it adds a shortcut path, not less capacity or a different normalization. Predict
this improves or is neutral for `classification_accuracy_val`. Same-stage residual only (`x +
Conv(x)`-style) — no encoder-to-decoder path, respects the no-skip-connections rule.

## Implementation
New class `AutoEncoder3D_AsymResSeparableV2_SELateResBottleneck` in `src/models/ae_models.py`
(model_name `AE3dAsymResSeparableV2SELateResBottleneck`), copied from
`AutoEncoder3D_AsymResSeparableV2_SELate` (parent `761cab78`) with exactly one change:
`self.bottleneck_conv = nn.Sequential(...)` (two plain conv+InstanceNorm+ReLU) replaced by
`self.bottleneck_conv = ResConv3DBlock(64, 128, downsample=False)` — an already-existing, already-used
block elsewhere in this file (e.g. `AutoEncoder3D_AsymResidual`), reused here, not a new block
definition. Output shape identical (128x2x8x8), so `final_down`/`fc_enc`/`fc_dec`
(flattened_size=2048) are unchanged. Same SE placement (se3/se4 only), enc4 dilation=1, same
hyperparameters (`lr=1e-4`, `weight_decay=0`, `dropout_rate=0`, `noise_std=0`, `patience=20`) as the
champion. Verified the new class builds, forward-passes, and produces the expected (1,20) latent:
param count 1,137,801 vs champion `761cab78`'s 1,129,481 (delta +8,320 = exactly the 1x1x1 shortcut
conv's params, 64*128+128, confirming no other capacity change).

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