---
model_name: AE3dAsymResSeparableV2
summary: Halve all channel widths throughout encoder/bottleneck/decoder (8/16/32/64/128 -> 4/8/16/32/64, flattened 2048->1024), opposite direction from the two failed capacity-increase trials b606a10f and 0cadad28
parent: 3aa0388f
id: null
status: draft
verdict: null
created_at: null
metric:
  primary: {name: classification_accuracy_val, value: null, direction: maximize}
---

# Trial <id> — AE3dAsymResSeparableV2 — <verdict>

<!-- ===== written BEFORE the run (agent) ===== -->

## Hypothesis
Two consecutive architecture trials this campaign added capacity and both failed:
b606a10f (whole-network doubling, -0.05 vs champion) and 0cadad28 (encoder-only
doubling, -0.0833 vs champion, worse — and its near-flat validation_R2_mean plus
ballooned per-seed accuracy spread pointed to plain overfitting/optimization variance
from too many parameters for 100 training volumes, not a bottleneck-dilution artifact).
Both results are consistent with a single story: `AE3dAsymResSeparableV2` at its
champion width is already at or past the capacity this data volume supports for the
classification task, and adding parameters only adds variance. This trial tests the
direct, opposite prediction: if the model is already over-parameterized for n=100, then
HALVING every channel width should reduce overfitting/variance and match or improve
classification_accuracy_val relative to champion 3aa0388f (0.6083), and should show
LESS per-seed spread than the champion's already-tight 0.575-0.625 band (a smaller,
more constrained network should be less sensitive to seed-specific initialization/
overfitting than the widened variants were). validation_R2_mean is expected to drop
some (less capacity to reconstruct fine detail) but the campaign's own evidence so far
(trial 3aa0388f's dropout finding, and now two capacity trials) is that
classification_accuracy_val and reconstruction fidelity are not the same objective, so a
lower R2 alongside a flat-or-better accuracy would further confirm that story.

## Implementation
In `src/models/ae_models.py`, class `AutoEncoder3D_AsymResSeparableV2`: every channel
count halved end-to-end — `enc1` 1->4 (was 1->8), `enc2` 4->8 (was 8->16), `enc3` 8->16
(was 16->32), `enc4` 16->32 (was 32->64), `bottleneck_conv` 32->64->64 (was 64->128->128),
`final_down` 64->64 (was 128->128), `feature_shape`=(64,1,4,4), `flattened_size`=1024
(was 2048). Decoder mirrored symmetrically: `dec1` 64->32, `dec2` 32->16, `dec3` 16->8,
`dec4_conv` 8->4, `final_conv` 4->1. Same block types (`ResSeparableConv3DBlock`,
`SeparableConv3DBlock`, `ResUpSeparableConv3DBlock`), same depth, same pooling
(`z_pool3`, `z_up`). No skip connections. In `configs/autoencoder.yaml` (only field
touched): none — all HPs (`lr=5e-4`, `weight_decay=1e-5`, `dropout_rate=0.3`,
`noise_std=0.0`, `patience=20`) kept identical to champion, isolating the
capacity-reduction change alone.

<!-- ===== written AFTER the run ===== -->

## Results
<!-- Filled automatically by the driver — leave empty. -->

## Training Dynamics
<!-- Agent, after the run: stability, convergence speed, spikes, plateau, early stopping. -->

## Conclusion
<!-- Agent, after the run: did the hypothesis hold? Mechanistic explanation of why it
     worked or failed — not just the numbers. -->
