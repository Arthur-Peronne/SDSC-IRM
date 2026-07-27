---
model_name: AE3dAsymDilatedAttention
summary: Replace champion's ResSeparableConv3DBlock encoder with DilatedAttentionConv3DBlock (dilation 1→2→4) + SE attention to capture multi-scale cardiac context; decoder identical to champion
parent: 3aa0388f
id: null
status: draft
verdict: null
created_at: null
metric:
  primary: {name: avg_validation_R2_mean, value: null, direction: maximize}
---

# Trial <id> — <model_name> — <verdict>

## Hypothesis
Four consecutive trials (b606a10f, 0cadad28, 001b5a08, e4e99f58) have failed to improve classification_accuracy_val by manipulating channel width (×2, ×0.5, encoder-only) or weight_decay (100×). This pattern strongly suggests that capacity and L2 regularization are not the levers that can improve this metric — the champion's architecture is already well-tuned for its current design.

This trial proposes a fundamentally different encoder mechanism: **dilated convolutions with Squeeze-and-Excitation attention**. Dilated convolutions increase the receptive field exponentially (1→2→4) without losing spatial resolution, allowing the network to capture multi-scale cardiac context — from local tissue textures to global heart geometry — in a single forward pass. SE attention performs channel-wise feature recalibration, prioritizing features that encode cardiac structures over background noise.

The key mechanistic prediction is that the latent representation will be more discriminative for the classification task because:
1. Multi-scale context provides richer structural information than single-scale convolutions
2. Attention focuses the bottleneck on diagnostically relevant features
3. The combination captures both "what" (cardiac structures) and "where" (spatial relationships)

I predict classification_accuracy_val will exceed the champion's 0.6083.

## Implementation
In `src/models/ae_models.py`:
- New class `AutoEncoder3D_AsymDilatedAttention` inserted immediately above `build_autoencoder`
- Encoder: `DilatedAttentionConv3DBlock` (dilation=1,2,4) for enc1-enc3, `DilatedConv3DBlock` (dilation=1) for enc4
- Same V4 asymmetric pooling: `pool1=(1,2,2)`, `z_pool3=(2,1,1)`
- Same bottleneck: `Conv3d×2` → `flatten` → `FC(2048, latent_dim)`
- Decoder: identical to champion (`ResUpSeparableConv3DBlock` × 3 + `ResSeparableConv3DBlock`)
- No skip connections

In `configs/autoencoder.yaml`:
- `model_name: "AE3dAsymDilatedAttention"` (new model)
- All HPs identical to champion: `lr=5e-4`, `weight_decay=1e-5`, `dropout_rate=0.3`, `noise_std=0.0`, `patience=20`

This isolates the encoder mechanism change alone against the exact champion HPs.