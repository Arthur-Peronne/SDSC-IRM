---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2SELateSE5
summary: "Add SE gate (se5) on bottleneck_conv's 128-channel output, right before flatten/fc_enc — extends the campaign's only 2-for-2 mechanism (SE gating) to the most classification-proximal feature map"
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
Three consecutive trials targeting `bottleneck_conv`'s internal structure have now closed that avenue
(capacity reduction `1ee5e03c`: failed; normalization swap `2647e285`: near-miss; residual add
`c239e8d5`: failed) — pivoting away from further `bottleneck_conv` structural changes per that trial's
own conclusion. Instead, extending this campaign's one mechanism with a clean track record: SE channel
gating (`a581f44e` CHAMPION: full-network SE; `761cab78` CHAMPION: SE restricted to enc3/enc4;
`09415e52` FAILURE: SE on enc4 only, showing enc3's gate matters too). SE has never been applied to
`bottleneck_conv`'s 128-channel output — the feature map immediately flattened and linearly projected
into the latent that feeds the logistic classifier, i.e. the single most classification-proximal point
in the whole network. If channel-wise recalibration helps at 32/64 channels (enc3/enc4), it should
plausibly help at least as much right before the projection that matters most for the judge metric,
by letting the network downweight reconstruction-only channels relative to group-relevant ones at the
last possible point. This is a continuation of the one validated-positive mechanism, at a genuinely new
location, not a repeat of an already-tested placement.

## Implementation
New class `AutoEncoder3D_AsymResSeparableV2_SELateSE5` in `src/models/ae_models.py` (model_name
`AE3dAsymResSeparableV2SELateSE5`), copied from `AutoEncoder3D_AsymResSeparableV2_SELate` (parent
`761cab78`) with exactly one addition: a new `self.se5 = SEBlock3D(128)` applied to
`bottleneck_conv`'s output, right before `final_down` (`x = self.bottleneck_conv(x); x = self.se5(x);
x = self.final_down(x)`). `bottleneck_conv` itself is left exactly as in the champion — plain two-conv
`InstanceNorm3d` stack, no capacity/normalization/residual change — isolating this trial to the SE
question alone, distinct from the three already-closed `bottleneck_conv`-internal trials. Everything
else (se3/se4 placement, enc4 dilation=1, `final_down`/`fc_enc`/`fc_dec`, hyperparameters `lr=1e-4`,
`weight_decay=0`, `dropout_rate=0`, `noise_std=0`, `patience=20`) is identical to the champion. No
encoder-to-decoder path — respects the no-skip-connections rule. Verified the new class builds,
forward-passes, and produces the expected (1,20) latent: param count 1,131,529 vs champion
`761cab78`'s 1,129,481 (delta +2,048 = SE block's two bias-free Linear layers, 128*8 + 8*128,
confirming no other capacity change).

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