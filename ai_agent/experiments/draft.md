---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2SELateDilatedEnc4      # the AE architecture trained (e.g. AE3dDilatedAttention). Descriptive, NOT the id.
summary: "Give champion's enc4 dilation=2 for a wider receptive field right before the bottleneck"         # one-line description of the change (becomes the CSV modification_description)
parent: 761cab78         # lineage: `id` of the trial whose CODE this one branched FROM. null for roots.

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
Fourth distinct architectural family this campaign (after SE placement, pooling mechanism, and FC
bottleneck depth — all now closed or falsified directions; also distinct from the failed
regularization HP check). Champion `AE3dAsymResSeparableV2SELate`'s `enc4` (last encoder stage, 32
-> 64 channels) uses a plain `SeparableConv3DBlock` with `dilation=1`. I set `dilation=2`. Mechanism:
at enc4's 4x16x16 input resolution (after `z_pool3`), a dilation-2 3x3x3 kernel covers a much wider
effective receptive field than dilation=1, letting this stage relate more spatially distant regions
(e.g. septum vs. free wall, or apex-ward vs. base-ward slices) in a single convolution, right before
the volume is compressed into the bottleneck. `enc4` is exactly the stage the SE-placement ablation
(761cab78 vs 09415e52) showed carries real signal — testing whether giving it more spatial context on
top of that channel recalibration helps further, rather than just recalibrating the context it
already has. `SeparableConv3DBlock` already exposes a `dilation` parameter, so this needs no new
block class and changes nothing else about the model. Same SE placement, same hyperparameters
(`lr=1e-4`, `weight_decay=0`, `dropout_rate=0`, `noise_std=0`, `patience=20`) as the champion.

## Implementation
New class `AutoEncoder3D_AsymResSeparableV2_SELateDilatedEnc4` in `src/models/ae_models.py` (model_name
`AE3dAsymResSeparableV2SELateDilatedEnc4`), copied from `AutoEncoder3D_AsymResSeparableV2_SELate`
(current champion) with exactly one line changed: `enc4 = SeparableConv3DBlock(32, 64,
downsample=True)` -> `SeparableConv3DBlock(32, 64, dilation=2, downsample=True)`. Padding inside
`SeparableConv3DBlock` already scales with `dilation` (`padding=dilation`), so spatial output shape
is unchanged (64x2x8x8) — only the receptive field grows. No encoder-to-decoder path — respects the
no-skip-connections rule in `program.md`. Verified the new class builds, forward-passes, and produces
a (1,20) latent with the expected (near-identical) parameter count before launching the trial.

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