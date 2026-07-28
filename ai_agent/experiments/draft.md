---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2SE      # the AE architecture trained (e.g. AE3dDilatedAttention). Descriptive, NOT the id.
summary: "Add SE channel recalibration (SEBlock3D) after every encoder stage of the champion AsymResSeparableV2"         # one-line description of the change (becomes the CSV modification_description)
parent: ff5882ad         # lineage: `id` of the trial whose CODE this one branched FROM. null for roots.

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
Fuses two prior ideas: the champion `AE3dAsymResSeparableV2` (separable-residual encoder, best
reconstruction R2 in the archived architecture campaign) and `AE3dAttention`'s SE (Squeeze-and-
Excitation) channel recalibration. I add an `SEBlock3D` after every encoder stage (enc1-enc4),
before the bottleneck FC. Mechanism: SE lets the network learn, per-sample, which feature channels
matter (e.g. cardiac-structure channels vs. background/noise channels) before those channels are
irreversibly compressed into the 20-d latent. For classification accuracy specifically (not just
reconstruction), I predict this produces a latent that encodes more group-discriminative signal,
since irrelevant/noisy channels get down-weighted before compression rather than diluting the fixed
20-d budget. All hyperparameters kept identical to the BASELINE trial (`lr=1e-4`, `weight_decay=0`,
`dropout_rate=0`, `noise_std=0`, `patience=20`) so any effect is attributable to the architecture
change alone.

## Implementation
New class `AutoEncoder3D_AsymResSeparableV2_SE` in `src/models/ae_models.py` (model_name
`AE3dAsymResSeparableV2SE`), copied from `AutoEncoder3D_AsymResSeparableV2` with one addition: an
`SEBlock3D` inserted right after each encoder block, before pooling/the next stage
(enc1->se1->pool1, enc2->se2, enc3->se3->z_pool3, enc4->se4->bottleneck_conv). Decoder is untouched.
`se1` uses `reduction=2` (not the class default `16`) because `8 // 16 == 0` would collapse the SE
gate for the first stage's 8 channels to a constant no-op; `se2/se3/se4` (16/32/64 channels) use the
default `reduction=16`, consistent with the codebase's existing `AttentionConv3DBlock` usage.
SE operates channel-wise within each stage (no encoder-to-decoder path) — respects the
no-skip-connections rule in `program.md`. Verified the new class builds, forward-passes, and
produces a (1,20) latent with no shape/warning issues before launching the trial.

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