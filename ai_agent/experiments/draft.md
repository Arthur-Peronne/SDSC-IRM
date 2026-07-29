---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2SELateEnc3OnlyDilatedEnc4
summary: "Fuse champion ac5057cf (se3-only) with 287085ec's enc4 dilation=2 (previously a tie on the se3+se4 lineage) — testing directly rather than assuming after the GroupNorm fusion failed to compose"
parent: ac5057cf

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
Fusion of two prior ideas on the new leaner base. Parent 1: `ac5057cf` (CHAMPION, se3-only). Parent 2:
`287085ec` (CHAMPION-by-tie, enc4 dilation=2, tested on the `761cab78`/se3+se4 lineage — a
statistical tie, its own conclusion speculating enc4's tiny 4×16×16 input leaves little genuinely new
receptive field to gain). The last fusion attempt (`b02b8293`, se3-only + bottleneck GroupNorm) failed
to compose — its conclusion warned against assuming independently-tested changes combine, especially
when one touches channel/normalization structure. Dilation is a different kind of mechanism (pure
receptive-field change, no covariance or channel-grouping assumption), so there's less a priori reason
to expect the same failure mode, but testing directly rather than assuming either way. Question: with
the redundant `se4` gate now removed (`ac5057cf`), does `enc4`'s wider receptive field remain neutral
(as it was tied on the se3+se4 lineage), or does it interact differently now that enc4's output isn't
immediately re-gated by a second SE block competing for the same channel information? No
encoder-to-decoder path — respects the no-skip-connections rule.

## Implementation
New class `AutoEncoder3D_AsymResSeparableV2_SELateEnc3OnlyDilatedEnc4` in `src/models/ae_models.py`
(model_name `AE3dAsymResSeparableV2SELateEnc3OnlyDilatedEnc4`), copied from
`AutoEncoder3D_AsymResSeparableV2_SELateEnc3Only` (parent `ac5057cf`) with exactly one change: `enc4 =
SeparableConv3DBlock(32, 64, downsample=True)` -> `SeparableConv3DBlock(32, 64, dilation=2,
downsample=True)` — the exact change `287085ec` tested on the se3+se4 lineage. Everything else
(se3-only placement, `bottleneck_conv` unchanged/plain InstanceNorm3d, hyperparameters `lr=1e-4`,
`weight_decay=0`, `dropout_rate=0`, `noise_std=0`, `patience=20`) identical to `ac5057cf`. Verified the
new class builds, forward-passes, and produces the expected (1,20) latent: param count identical to
`ac5057cf`'s 1,128,969 (dilation changes receptive field, not parameter count).

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