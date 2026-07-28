---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2SELateBottleneckGN
summary: "Replace bottleneck_conv's two InstanceNorm3d with GroupNorm(8,128) to preserve inter-channel relative scale before the FC projection"
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
Sixth distinct architectural mechanism this campaign (after SE placement, CBAM spatial attention,
learnable strided downsampling, nonlinear FC bottleneck, dilation, and bottleneck layer count — all
now closed or falsified). `bottleneck_conv`'s two `InstanceNorm3d(128)` layers each normalize every
channel independently (own per-instance mean/std only), discarding any relative-scale relationship
between the 128 channels — exactly the kind of cross-channel information the SE-placement result
(`761cab78` vs `09415e52`) showed carries real signal at this depth (se3/se4 gate channels relative to
each other, and removing that gating from earlier, less semantic stages helped). I replace both
`InstanceNorm3d(128)` with `GroupNorm(num_groups=8, num_channels=128)` (16 channels/group), which
normalizes each group of channels jointly, preserving relative inter-channel scale within a group
while remaining batch-size independent (unaffected by `batch_size=1`, unlike BatchNorm — so it's a
like-for-like swap, no training-loop changes needed). Predicting this helps or is neutral for
`classification_accuracy_val`, since it lets more inter-channel information reach the flatten+FC step
that feeds the classifier, without touching capacity (params +512 only, from GroupNorm's learnable
affine vs InstanceNorm3d's default `affine=False`).

## Implementation
New class `AutoEncoder3D_AsymResSeparableV2_SELateBottleneckGN` in `src/models/ae_models.py`
(model_name `AE3dAsymResSeparableV2SELateBottleneckGN`), copied from
`AutoEncoder3D_AsymResSeparableV2_SELate` (parent `761cab78`, deliberately not `287085ec`'s dilated
enc4 variant — that trial's dilation was a statistical tie, not a validated gain, so I avoid
compounding an unvalidated change with this new one) with exactly one change: both
`nn.InstanceNorm3d(128)` inside `bottleneck_conv` replaced by `nn.GroupNorm(8, 128)`. Everything else
—SE placement (se3/se4 only), enc4 dilation=1, `final_down`, `fc_enc`/`fc_dec` (flattened_size=2048
unchanged), hyperparameters (`lr=1e-4`, `weight_decay=0`, `dropout_rate=0`, `noise_std=0`,
`patience=20`)—is identical to the champion. `SeparableConv3DBlock`'s own `InstanceNorm3d` layers
(used by enc4) are untouched, isolating the effect to the last two conv layers before the latent
projection. No encoder-to-decoder path — respects the no-skip-connections rule. Verified the new
class builds, forward-passes, and produces the expected (1,20) latent: param count 1,129,993 vs
champion `761cab78`'s 1,129,481 (delta +512 = 2 GroupNorm layers × 128 channels × 2 affine params,
exactly the expected learnable-affine cost, confirming no other capacity change).

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