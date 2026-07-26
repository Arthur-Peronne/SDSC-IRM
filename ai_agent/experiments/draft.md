---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2
summary: "Increase bottleneck dropout_rate 0.1->0.3, reverting noise_std to 0.0 (trial f826eedd's failed direction), branching from champion 725420c7"
parent: 725420c7

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
Trial f826eedd (FAILURE) showed that regularizing via input-level noise (noise_std=0.05)
destroys fine anatomical detail the classifier depends on: val R2 collapsed to 0.41 and
accuracy dropped to 0.4167 (vs champion 0.5917), with all 3 seeds' val loss peaking at
epoch 2 then degrading — a sign the perturbation was applied at the wrong place in the
pipeline (raw input voxels) for this small-data (100 train volumes), batch_size=1 regime.
This trial tests generalization regularization at a different, more targeted location:
`dropout_rate`, which in `AE3dAsymResSeparableV2` (confirmed by reading `ae_models.py`
lines 1832-1833/1860 and 1866) is applied only on the flattened 2048-d bottleneck vector,
immediately before `fc_enc` (encode side) and immediately after `fc_dec` (decode side) —
i.e. directly on/around the very representation `z` that feeds the downstream classifier,
not on the raw input volume. Mechanistically, this should force redundancy in *which*
bottleneck features the network relies on to reconstruct (preventing over-reliance on a
few fragile feature-vector coordinates) without ever corrupting the input signal itself,
so it should regularize without the same destructive effect on fine anatomical boundaries.
Champion trial 725420c7 already used a mild dropout_rate=0.1; I predict a further
increase to 0.3 (mild-to-moderate for a bottleneck-only dropout) improves generalization
and therefore classification_accuracy_val above 0.5917, while degrading much less (if at
all) than the noise trial did.

## Implementation
In `configs/autoencoder.yaml` (the only mutable file touched this trial), branching from
champion 725420c7's config:
- `dropout_rate: 0.1 -> 0.3` (bottleneck dropout only, see Hypothesis for exact location)
- `noise_std: 0.05 -> 0.0` (revert trial f826eedd's failed change back to champion value —
  this trial isolates the dropout effect alone, not a combination with noise)
- Unchanged from champion: `weight_decay=1e-5`, `lr=5e-4`, `patience=20`,
  `latent_dimensions=20`, `model_name=AE3dAsymResSeparableV2`. No architecture file
  touched this trial (isolating one HP change keeps the comparison clean); architecture
  changes (`src/models/ae_models.py`, newly opened this campaign) are deferred to a later
  trial once the HP-tuning direction is better characterized. No skip connections.

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