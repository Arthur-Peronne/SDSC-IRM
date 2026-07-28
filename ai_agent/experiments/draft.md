---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2SELate      # the AE architecture trained (e.g. AE3dDilatedAttention). Descriptive, NOT the id.
summary: "Ablation: keep champion's SE only on enc3/enc4 (remove se1/se2)"         # one-line description of the change (becomes the CSV modification_description)
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
Ablation of the champion `AE3dAsymResSeparableV2SE` (trial a581f44e): keep SE recalibration only on
enc3/enc4 (32/64 channels), remove it from enc1/enc2 (8/16 channels). Mechanism: enc1/enc2 likely
still encode low-level, generic features (edges, local texture) shared across all patients
regardless of group — a channel gate there has little group-relevant signal to recalibrate, and may
even inject noise this early via its own learned parameters. enc3/enc4 are closer to the bottleneck
and more likely to encode semantic, higher-level patterns where channel reweighting has real
leverage for what ends up in the 20-d latent. If the champion's gain is concentrated in the late
gates, this should match or beat a581f44e with fewer parameters; if it's spread across all stages,
this should underperform it. Same hyperparameters as prior trials (`lr=1e-4`, `weight_decay=0`,
`dropout_rate=0`, `noise_std=0`, `patience=20`).

## Implementation
New class `AutoEncoder3D_AsymResSeparableV2_SELate` in `src/models/ae_models.py` (model_name
`AE3dAsymResSeparableV2SELate`), copied from `AutoEncoder3D_AsymResSeparableV2_SE` (current
champion) with `se1` and `se2` removed — `enc1`/`enc2` now feed directly into `pool1`/`enc3` with no
gating in between. `se3`/`se4` kept exactly as in the champion. Decoder unchanged. Same-stage gating
only, no encoder-to-decoder path — respects the no-skip-connections rule in `program.md`. Verified
the new class builds, forward-passes, and produces a (1,20) latent before launching the trial.
Note: this is a distinct hypothesis from the CBAM trial (00c71333, FAILURE) — it changes *where SE
gates apply*, not *what kind of gate* is added — so it is not a repeat of a failing idea.

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