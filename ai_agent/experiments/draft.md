---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2SELateDilatedEnc3
summary: "Give champion's enc3 dilation=2 (unlike the neutral enc4 dilation trial, enc3 operates at a much larger 16x32x32 pre-downsample resolution with real receptive field to gain)"
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
Four consecutive trials targeting the bottleneck area (`1ee5e03c`, `2647e285`, `c239e8d5`, `125727a1`)
all landed within noise of the champion — deliberately moving to a different location. Revisiting
dilation, but at `enc3` instead of `enc4`. `287085ec` (enc4 dilation=2) was a statistical tie with
`761cab78`, and its own conclusion argued the mechanism was likely inert rather than harmful: `enc4`
operates on an already-tiny 4×16×16 input, where a wider receptive field may have little genuinely new
context to cover. `enc3` is structurally different — its convs run at 16×32×32 (pre its own
downsample), a much larger spatial extent with real room for a wider receptive field to matter, and
it's exactly the stage the SE-ablation (`761cab78` vs `09415e52`) showed carries real classification
signal. Predicting dilation=2 at `enc3` has a materially different (not necessarily better, but not
inert) effect on `classification_accuracy_val` than the enc4 version did, by letting this stage relate
more spatially distant regions (e.g. septum vs. free wall) at a resolution where that's still spatially
meaningful. Same-stage dilation only — no encoder-to-decoder path, respects the no-skip-connections
rule.

## Implementation
New class `AutoEncoder3D_AsymResSeparableV2_SELateDilatedEnc3` in `src/models/ae_models.py`
(model_name `AE3dAsymResSeparableV2SELateDilatedEnc3`), copied from
`AutoEncoder3D_AsymResSeparableV2_SELate` (parent `761cab78`) with exactly one change: `enc3 =
ResSeparableConv3DBlock(16, 32, downsample=True)` -> `ResSeparableConv3DBlock(16, 32, dilation=2,
downsample=True)`. This required adding an optional `dilation=1` parameter to the shared
`ResSeparableConv3DBlock` class (applied to both depthwise convs, `padding=dilation`, matching the
pattern `SeparableConv3DBlock` already used for the `enc4` dilation trial) — default value preserves
byte-for-byte identical behavior for every other existing caller of that block (`enc1`, `enc2`, and
`dec4_conv` in this same class, plus other archived models). Padding scales with dilation so spatial
output shape is unchanged (32×8×16×16 after `enc3`'s own downsample) — only the receptive field grows.
Everything else (SE placement se3/se4, enc4 dilation=1, bottleneck_conv unchanged, hyperparameters
`lr=1e-4`, `weight_decay=0`, `dropout_rate=0`, `noise_std=0`, `patience=20`) identical to the champion.
Verified the new class builds, forward-passes, and produces the expected (1,20) latent with param
count identical to champion `761cab78`'s 1,129,481 (dilation changes receptive field, not parameter
count).

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