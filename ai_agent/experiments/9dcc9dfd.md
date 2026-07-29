---
model_name: AE3dAsymResSeparableV2SELateDilatedEnc3
summary: Give champion's enc3 dilation=2 (unlike the neutral enc4 dilation trial,
  enc3 operates at a much larger 16x32x32 pre-downsample resolution with real receptive
  field to gain)
parent: 761cab78
id: 9dcc9dfd
status: completed
verdict: FAILURE
created_at: '2026-07-29T00:44:10+00:00'
metric:
  primary:
    name: classification_accuracy_val
    value: 0.666667
    direction: maximize
---

# Trial 9dcc9dfd — AE3dAsymResSeparableV2SELateDilatedEnc3 — FAILURE

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
- **accuracy_test per run:** 0: 0.675000 | 1: 0.675000 | 2: 0.650000
- **classification_accuracy_val:** 0.666667
- **delta_vs_champion** (display only): -0.033333
- **validation_R2_mean** (mean, AE phase, non-decisional): 0.721108
- **AE MLflow Run IDs:** 8d90136ae5b24dcbb7eec0f725a390f1 be976bb5ed5c4a31b323fd9089c355bd 4f4a255f708c421081f481bd049ad27d
- **Classification MLflow Run IDs:** e45e7957edd3454897e5b491c4ebe057 f168a169efe64be6ba3b6556b70146bc ae56a1065dcd401e94bd72e329653b7b

## Training Dynamics
<!-- Agent, after the run: stability, convergence speed, spikes, plateau, early stopping. -->

## Conclusion
<!-- Agent, after the run: did the hypothesis hold? Mechanistic explanation of why it
     worked or failed — not just the numbers. -->