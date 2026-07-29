---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2SELate
summary: "HP tuning on champion architecture (no code change): lr 1e-4 -> 5e-5, testing whether a lower learning rate reduces the convergence-speed/R2 variance seen across recent architecture trials"
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
Five consecutive architecture trials (`1ee5e03c` through `9dcc9dfd`) all failed to beat the champion,
several within noise but two (`c239e8d5`, `9dcc9dfd`) showing genuinely wider seed-to-seed variance
and slower/less stable convergence than the champion itself (e.g. `9dcc9dfd` seed 0 converged shallow
and fast while seeds 1/2 needed ~35 epochs; `c239e8d5` seed 0 needed 87 epochs, the slowest of the
campaign). `lr` has not been touched at all this campaign — every trial so far used the inherited
`lr=1e-4`. Per `program.md`, hyperparameters are explicitly open alongside architecture, and with
architecture ideas near this bottleneck/mid-network area showing diminishing, noise-dominated returns,
testing whether the champion architecture itself is under- or over-tuned on `lr` is a legitimate,
cheap (no code change) next step. Halving `lr` to `5e-5` should slow and smooth convergence, which I
predict either improves `classification_accuracy_val` (if the champion is presently in a slightly
noisy/overshooting regime) or is neutral — testing the champion's own tuning before spending more
trials on architecture variants of it.

## Implementation
No new class — this trial reuses the existing, already-registered `AutoEncoder3D_AsymResSeparableV2_SELate`
(model_name `AE3dAsymResSeparableV2SELate`, same code as parent `761cab78`/tied-champion `287085ec`'s
lineage, but explicitly the non-dilated version, since dilation was separately falsified at both
tested locations this campaign). Only `configs/autoencoder.yaml` changes: `lr: 1e-4 -> 5e-5`,
`model_name`/`models_list` set to `AE3dAsymResSeparableV2SELate` (was left on
`AE3dAsymResSeparableV2SELateDilatedEnc4` after the previous trial's auto-revert). All other
hyperparameters (`weight_decay=0`, `dropout_rate=0`, `noise_std=0`, `patience=20`) and the architecture
itself are byte-for-byte the champion's — isolating this trial to the single `lr` change.

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