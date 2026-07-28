---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2SECBAM      # the AE architecture trained (e.g. AE3dDilatedAttention). Descriptive, NOT the id.
summary: "Add CBAM spatial attention at the bottleneck (128x2x8x8), on top of champion's SE"         # one-line description of the change (becomes the CSV modification_description)
parent: a581f44e         # lineage: `id` of the trial whose CODE this one branched FROM. null for roots.

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
Builds on the current champion (`AE3dAsymResSeparableV2SE`, trial a581f44e), adding a CBAM-style
spatial attention gate on the bottleneck feature map (128x2x8x8, right after `bottleneck_conv`,
before `final_down`). SE already recalibrates *which channels* matter at every encoder stage; this
adds a complementary gate for *which spatial locations* matter, at the last and most compact feature
map before the volume is irreversibly flattened into the 20-d latent. Mechanism: cardiac
group-discriminative structure (septum thickness/shape, wall regions) is likely concentrated in a
subset of voxels rather than spread uniformly — suppressing background/less-informative locations
right before the one-shot spatial-to-vector compression should let more of that structure survive
into the fixed 20-d budget, on top of SE's channel-level gain. Same hyperparameters as the BASELINE/
SE trials (`lr=1e-4`, `weight_decay=0`, `dropout_rate=0`, `noise_std=0`, `patience=20`) to isolate
the architecture effect.

## Implementation
New class `AutoEncoder3D_AsymResSeparableV2_SE_CBAM` in `src/models/ae_models.py` (model_name
`AE3dAsymResSeparableV2SECBAM`), copied from `AutoEncoder3D_AsymResSeparableV2_SE` (the current
champion) with one addition: a new `SpatialAttention3D` module (channel-avg + channel-max pooled
into 2 maps -> 3x3x3 conv -> sigmoid -> multiplicative gate) applied to `bottleneck_conv`'s output,
before `final_down`. Kernel size 3 (not CBAM's usual 7) because the feature map's z-dimension is
only 2 voxels — a size-7 kernel there would be disproportionate. Everything else (encoder SE blocks,
decoder) unchanged. Attention is applied entirely within the bottleneck stage, on the
already-compressed 128-channel map — no encoder-to-decoder path, respects the no-skip-connections
rule in `program.md`. Verified the new class builds, forward-passes, and produces a (1,20) latent
with no shape/warning issues before launching the trial.

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