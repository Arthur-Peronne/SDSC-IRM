---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2SELateSimpleBottleneck
summary: "Reduce bottleneck_conv from two stacked 3x3x3 convs (64->128->128) to a single one (64->128), generalizing the SE-placement win (less unnecessary bottleneck-adjacent capacity)"
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
Fifth distinct architectural family this campaign. This campaign's single strongest validated result
so far is the SE-placement ablation (`761cab78` vs `09415e52`/parent `a581f44e`): removing SE gates
from enc1/enc2 (channel recalibration on generic low-level features) improved BOTH reconstruction and
classification over gating every stage — evidence that not all capacity near the bottleneck is useful,
and that removing the unhelpful part can help rather than just being neutral. I generalize that same
"less unnecessary bottleneck-adjacent capacity" hypothesis to a different component: `bottleneck_conv`
(currently two stacked, unregularized 3x3x3 convs, 64->128->128, right before the FC projection) may
likewise carry more capacity than this task needs, encouraging the classifier-relevant signal to get
diluted or overfit to reconstruction-only detail. I reduce it to a single 64->128 conv, predicting this
either improves or is neutral for `classification_accuracy_val`, mirroring the SE-ablation pattern,
while cutting ~440k parameters (~39% of the model: 1.13M -> 0.69M). This is a distinct mechanism from
SE placement, pooling method, FC depth, dilation, or regularization — all already tested this campaign.

## Implementation
New class `AutoEncoder3D_AsymResSeparableV2_SELateSimpleBottleneck` in `src/models/ae_models.py`
(model_name `AE3dAsymResSeparableV2SELateSimpleBottleneck`), copied from
`AutoEncoder3D_AsymResSeparableV2_SELate` (parent champion `761cab78`) with exactly one change:
`bottleneck_conv` drops its second `Conv3d(128,128,3,1,1) + InstanceNorm3d + ReLU` stage, keeping only
the first `Conv3d(64,128,3,1,1) + InstanceNorm3d + ReLU`. `final_down` (128->128, stride 2) is
unchanged, so the FC input shape stays identical (flattened_size=2048, `fc_enc`/`fc_dec` untouched).
Same SE placement (se3, se4 only, per champion), same hyperparameters (`lr=1e-4`, `weight_decay=0`,
`dropout_rate=0`, `noise_std=0`, `patience=20`) as the champion. No encoder-to-decoder path — respects
the no-skip-connections rule in `program.md`. Verified the new class builds, forward-passes, and
produces the expected (1,20) latent: param count 686,985 vs champion's 1,129,481 (delta -442,496,
consistent with removing exactly one 128->128 3x3x3 conv + InstanceNorm).

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