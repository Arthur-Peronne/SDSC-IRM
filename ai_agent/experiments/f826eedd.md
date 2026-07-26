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
All 3 seeds show the same pathological pattern: val loss reaches its minimum at
**epoch 2** and then increases almost monotonically (with noisy fluctuations) through
epoch 22, where the LR-scheduler-plus-patience combination (patience=20, halving LR
every 5 stale epochs) triggers early stopping — versus 52 epochs / best-epoch=32 in the
noise-free baseline (725420c7). Train loss keeps decreasing steadily the whole time
(0.0037 → 0.0020 by epoch 22), so this is not an optimizer failure — the network keeps
fitting *something* — but whatever it fits after epoch 2 stops generalizing to the
validation volumes. Reconstruction quality collapsed accordingly: validation R2 mean
0.41 (vs 0.71-0.75 baseline range), high std (0.43-0.48) indicating some patients
reconstructed reasonably and others very poorly, not a uniform degradation.

## Conclusion
Hypothesis **did not hold** — noise_std=0.05 hurt rather than helped generalization,
both on the AE's own reconstruction (R2 0.41 vs 0.71-0.75) and on downstream
classification accuracy (0.4167 vs 0.5917, -0.175, a FAILURE by a wide margin, well
past the noise floor of ~0.03-0.04). Mechanistic read: at batch_size=1 with only 100
training volumes, adding 5%-of-range Gaussian noise to every voxel of the input is a
much larger perturbation relative to this model's effective capacity/data budget than
anticipated — the val-loss-minimum-at-epoch-2 pattern (identical across all 3 seeds)
suggests the noise pushed the optimization landscape such that the very first
few gradient steps found the best generalizing point reachable, after which further
training over-fits the noise pattern itself rather than the denoising task. This
contradicts the (R2-only-campaign) intuition that noise_std is a "free" generalization
knob — here it actively destroys the fine anatomical boundaries the classifier depends
on, exactly the risk flagged in the Hypothesis. Actionable takeaway for later trials:
if noise-based regularization is revisited, it must be far weaker (order 0.005-0.01) and
probably paired with more patience/epochs, not applied at this magnitude; for the
immediate next trial, abandon this direction and branch again from 725420c7's proven
baseline rather than push noise further.