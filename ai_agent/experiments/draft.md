---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2SELateEnc3OnlyBottleneckGN
summary: "Fuse new champion ac5057cf (se3-only) with near-miss 2647e285's bottleneck GroupNorm — both independently improved R2, testing whether they compose on accuracy"
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
Fusion of two prior ideas. Parent 1: `ac5057cf` (new CHAMPION, se3-only — dropping se4 from
`761cab78` beat every other SE-placement combination, 0.7167, AND gave this campaign's best-ever R²
0.7432). Parent 2: `2647e285` (near-miss FAILURE on top of `761cab78` — `bottleneck_conv`'s two
`InstanceNorm3d` replaced by `GroupNorm(8,128)`, gave the previous-best R² 0.7419 but only
0.6917 accuracy, a -0.0083 near-tie). Both changes independently pushed reconstruction quality up;
`2647e285`'s conclusion explicitly speculated GroupNorm's extra information might be
reconstruction-relevant rather than classification-relevant on its own, but that test was run on the
`761cab78` architecture (with se4 still present). Testing whether GroupNorm's benefit transfers
differently on top of the leaner, better `ac5057cf` encoder — the two changes touch non-overlapping
locations (SE placement in the encoder vs. `bottleneck_conv`'s normalization), so there's no obvious
reason they should cancel out, and stacking two independently-R²-positive changes is a reasonable next
step now that a real (not just tied) champion exists to build on. No encoder-to-decoder path —
respects the no-skip-connections rule.

## Implementation
New class `AutoEncoder3D_AsymResSeparableV2_SELateEnc3OnlyBottleneckGN` in `src/models/ae_models.py`
(model_name `AE3dAsymResSeparableV2SELateEnc3OnlyBottleneckGN`), copied from
`AutoEncoder3D_AsymResSeparableV2_SELateEnc3Only` (parent `ac5057cf`) with exactly one addition: both
`nn.InstanceNorm3d(128)` inside `bottleneck_conv` replaced by `nn.GroupNorm(8, 128)` — the exact same
change `2647e285` tested on `761cab78`. Everything else (se3-only placement, enc4 dilation=1,
`final_down`/`fc_enc`/`fc_dec`, hyperparameters `lr=1e-4`, `weight_decay=0`, `dropout_rate=0`,
`noise_std=0`, `patience=20`) identical to `ac5057cf`. No encoder-to-decoder path — respects the
no-skip-connections rule. Verified the new class builds, forward-passes, and produces the expected
(1,20) latent: param count 1,129,481 vs `ac5057cf`'s 1,128,969 (delta +512 = GroupNorm's learnable
affine params, matching `2647e285`'s exact delta over its own parent, confirming no other capacity
change).

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