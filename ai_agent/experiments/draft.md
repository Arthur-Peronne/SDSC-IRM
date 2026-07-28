---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2SELateStrided      # the AE architecture trained (e.g. AE3dDilatedAttention). Descriptive, NOT the id.
summary: "Replace champion's 2 explicit anisotropic MaxPools with learnable depthwise strided convs"         # one-line description of the change (becomes the CSV modification_description)
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
New architectural family (not another SE-placement variant — that direction is now closed after 3
trials: a581f44e, 761cab78, 09415e52). Champion `AE3dAsymResSeparableV2SELate`'s two explicit
anisotropic poolers (`pool1`: 32x128x128->32x64x64, `z_pool3`: 32x8x16x16->32x4x16x16) are
`nn.MaxPool3d`, a fixed, non-learnable rule that keeps only the single strongest activation per
window and discards the rest. I replace both with depthwise strided convolutions of the same
kernel/stride (channel count unchanged via `groups=channels`, so parameter cost stays negligible).
Mechanism: a learnable downsampling can combine values within each window instead of hard-selecting
one, potentially preserving fine structural gradients (e.g. wall-thickness variation) that a max
operation discards outright — detail that may carry group-discriminative signal even where it
doesn't change which single activation is the local max. Per-block internal downsampling (inside
`ResSeparableConv3DBlock`/`SeparableConv3DBlock`, shared with other architectures) is left as
MaxPool — only the two standalone ops unique to this model are touched, to keep the change
contained. SE placement (se3, se4) and all hyperparameters (`lr=1e-4`, `weight_decay=0`,
`dropout_rate=0`, `noise_std=0`, `patience=20`) kept identical to the champion.

## Implementation
New class `AutoEncoder3D_AsymResSeparableV2_SELateStrided` in `src/models/ae_models.py` (model_name
`AE3dAsymResSeparableV2SELateStrided`), copied from `AutoEncoder3D_AsymResSeparableV2_SELate`
(current champion) with exactly two lines changed: `pool1` becomes
`nn.Conv3d(8, 8, kernel_size=(1,2,2), stride=(1,2,2), groups=8)` and `z_pool3` becomes
`nn.Conv3d(32, 32, kernel_size=(2,1,1), stride=(2,1,1), groups=32)` — same kernel/stride/channel
count as the MaxPools they replace, now with learnable weights. No encoder-to-decoder path —
respects the no-skip-connections rule in `program.md`. Verified the new class builds, forward-passes,
and produces a (1,20) latent before launching the trial.

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