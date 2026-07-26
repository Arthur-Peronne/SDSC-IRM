---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2      # the AE architecture trained (e.g. AE3dDilatedAttention). Descriptive, NOT the id.
summary: "Add denoising (noise_std=0.05) on top of trial 725420c7's light regularization"         # one-line description of the change (becomes the CSV modification_description)
parent: 725420c7          # lineage: `id` of the trial whose CODE this one branched FROM. null for roots.

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
Trial 725420c7 (BASELINE, weight_decay=1e-5, dropout_rate=0.1) showed stable training,
healthy reconstruction (val R2 mean=0.707) and classification accuracy (mean=0.5917,
tight seed spread) — confirming that config is a safe, non-destabilizing base to branch
from. This trial adds the other main generalization-oriented mechanism not yet tested:
input-noise denoising (`noise_std`). Mechanistically, corrupting the input volume with
small Gaussian noise before encoding (input clamped to [0,1]) and training the AE to
still reconstruct the clean target forces the encoder to capture more robust,
lower-frequency anatomical structure rather than exact per-voxel intensity idiosyncrasies
of the 100 training patients — structure that should transfer better to unseen
validation patients and therefore to the downstream classifier reading the latent code.
I predict classification_accuracy_val increases relative to trial 725420c7 (0.5917). I
picked noise_std=0.05 (5% of the normalized [0,1] intensity range) as a first, modest
value: strong enough to have a denoising effect but small enough not to destroy
fine anatomical boundaries the classifier's discriminative signal may depend on
(unlike the R2-only campaigns, here excessive corruption could hurt accuracy even if the
AE still "reconstructs" - the true target is downstream separability, not pixel fidelity).
weight_decay=1e-5 and dropout_rate=0.1 are kept unchanged from the proven-stable trial
725420c7, so any accuracy delta here isolates the marginal effect of adding noise_std on
top of that base, informing whether trial 3 should push noise further or abandon it.

## Implementation
In `configs/autoencoder.yaml` (the only mutable file), branching from trial 725420c7's
config:
- `noise_std: 0.0 -> 0.05` (denoising: Gaussian noise added to input at train time,
  clamped to [0,1], target remains the clean volume — implemented in
  `src/training/ae_training.py`, untouched/frozen this campaign)
- Unchanged from 725420c7: `weight_decay=1e-5`, `dropout_rate=0.1`, `lr=5e-4`,
  `patience=20`, `latent_dimensions=20`, `model_name=AE3dAsymResSeparableV2`. No skip
  connections, no architecture file touched.

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