---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2SELateEnc3OnlyReduction8
summary: "Refine the winning mechanism directly: se3's SE reduction 16 -> 8 (doubles the gate's hidden width, 2->4 units), after two fusion attempts on ac5057cf both failed"
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
Two fusion attempts on top of champion `ac5057cf` (bottleneck GroupNorm `b02b8293`, enc4 dilation
`0afac601`) both failed, and `0afac601`'s conclusion argued this base shouldn't be assumed compatible
with results from the older se3+se4 lineage — evaluate it directly instead. Rather than another
fusion or placement test, this trial refines the winning mechanism itself: `se3` currently uses
`SEBlock3D(32)` with the default `reduction=16`, squeezing the channel-gating function through a
2-unit hidden bottleneck (32/16=2) before re-expanding to 32 gate values — a very narrow function
space for deciding which of 32 channels matter. `reduction=8` doubles this to 4 units. Predicting a
2-unit gate may be under-parameterized for learning a classification-useful channel-importance
function, and a slightly wider gate could improve or be neutral for `classification_accuracy_val`,
with a negligible parameter cost (+128). No encoder-to-decoder path — respects the no-skip-connections
rule.

## Implementation
New class `AutoEncoder3D_AsymResSeparableV2_SELateEnc3OnlyReduction8` in `src/models/ae_models.py`
(model_name `AE3dAsymResSeparableV2SELateEnc3OnlyReduction8`), copied from
`AutoEncoder3D_AsymResSeparableV2_SELateEnc3Only` (parent `ac5057cf`) with exactly one change:
`self.se3 = SEBlock3D(32)` -> `SEBlock3D(32, reduction=8)`. Everything else (enc4 dilation=1,
`bottleneck_conv` unchanged, hyperparameters `lr=1e-4`, `weight_decay=0`, `dropout_rate=0`,
`noise_std=0`, `patience=20`) identical to the champion. Verified the new class builds,
forward-passes, and produces the expected (1,20) latent: param count 1,129,097 vs `ac5057cf`'s
1,128,969 (delta +128 = exactly the difference between `SEBlock3D(32, reduction=8)`'s 256 params and
`reduction=16`'s 128 params, confirming no other capacity change).

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