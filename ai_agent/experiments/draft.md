---
model_name: AE3dAsymResSeparableV2
summary: Double all channel widths throughout encoder/bottleneck/decoder (8/16/32/64/128 -> 16/32/64/128/256), same depth/blocks/pooling, branching from champion 3aa0388f
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
Champion 3aa0388f validated that bottleneck dropout_rate=0.3 is an effective regularizer
against overfitting on this small-data (100 train volumes) regime. This trial tests a
capacity change instead of a regularization change: doubling every channel width in
`AE3dAsymResSeparableV2` (encoder 8/16/32/64 -> 16/32/64/128, bottleneck conv 128 -> 256,
decoder mirrored, flattened bottleneck vector 2048 -> 4096) while keeping the exact same
block types, depth, and pooling schedule (no structural change, no skip connections
added). Mechanistically: the current network may be under-capacity for extracting the
fine-grained anatomical detail (chamber wall thickness, papillary muscle shape, etc.)
that discriminates ACDC groups, especially early in the encoder where channel count is
smallest (8 channels at full 32x128x128 resolution). More channels per layer let the
network represent a richer set of local features before any spatial downsampling
discards information. Given dropout=0.3 is already validated as a regularizer on the
(now larger, 4096-d) bottleneck, I predict the added capacity is usable rather than
purely overfit-prone, and that classification_accuracy_val improves above 0.6083. This
follows the architecture-priority directive: an architecture trial, not another HP
sweep, and it directly builds on 3aa0388f's validated dropout finding rather than
re-testing HPs in isolation.

## Implementation
In `src/models/ae_models.py`, class `AutoEncoder3D_AsymResSeparableV2`: every channel
count doubled end-to-end (enc1 1->16, enc2 16->32, enc3 32->64, enc4 64->128,
bottleneck_conv 128->256->256, final_down 256->256, decoder mirrored symmetrically
128->64->32->16, final_conv 16->1). `feature_shape`/`flattened_size` updated accordingly
(2048 -> 4096); `fc_enc`/`fc_dec` adapt automatically via `flattened_size`. No change to
block types (`ResSeparableConv3DBlock`, `SeparableConv3DBlock`, `ResUpSeparableConv3DBlock`),
depth, or pooling (`z_pool3`, `z_up`). No skip connections. In
`configs/autoencoder.yaml` (only field touched): none — all HPs (`lr=5e-4`,
`weight_decay=1e-5`, `dropout_rate=0.3`, `noise_std=0.0`, `patience=20`) kept identical
to champion 3aa0388f, isolating the capacity change alone.

<!-- ===== written AFTER the run ===== -->

## Results
<!-- Filled automatically by the driver — leave empty. -->

## Training Dynamics
<!-- Agent, after the run: stability, convergence speed, spikes, plateau, early stopping. -->

## Conclusion
<!-- Agent, after the run: did the hypothesis hold? Mechanistic explanation of why it
     worked or failed — not just the numbers. -->
