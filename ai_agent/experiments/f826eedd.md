---
model_name: AE3dAsymResSeparableV2
summary: Add denoising (noise_std=0.05) on top of trial 725420c7's light regularization
parent: 725420c7
id: f826eedd
status: completed
verdict: FAILURE
created_at: '2026-07-26T21:33:33+00:00'
metric:
  primary:
    name: classification_accuracy_val
    value: 0.416667
    direction: maximize
---

# Trial f826eedd — AE3dAsymResSeparableV2 — FAILURE

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
- **accuracy_test per run:** 0: 0.425000 | 1: 0.400000 | 2: 0.425000
- **classification_accuracy_val:** 0.416667
- **delta_vs_champion** (display only): -0.175000
- **validation_R2_mean** (mean, AE phase, non-decisional): 0.411139
- **AE MLflow Run IDs:** 0cd95653920649fdb364144fb70f3e0a 48793b5be349430f98a7720317d4504a 9b3110f1948640a1991977ddf737f63c
- **Classification MLflow Run IDs:** d8fecaea93ee4e5286a5bd495530ad80 c30e21a4f34f4c2cbc341b7fad9dc348 f4bb1ef0f7014b29acbaba8491b8d6a6

## Training Dynamics
<!-- Agent, after the run: stability, convergence speed, spikes, plateau, early stopping. -->

## Conclusion
<!-- Agent, after the run: did the hypothesis hold? Mechanistic explanation of why it
     worked or failed — not just the numbers. -->