---
model_name: AE3dAsymResSeparableBottleneckAttention
summary: Add SE attention to the bottleneck of champion architecture (encoder+decoder identical), targeting the latent representation directly
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
Six consecutive trials (b606a10f, 0cadad28, 001b5a08, e4e99f58, 25eae42a, 54805b0f) have failed by manipulating encoder architecture (channel width, weight_decay, dilated attention, standard+attention). The consistent failures suggest that encoder modifications alone are not the lever that improves classification_accuracy_val.

This trial pivots to a different part of the architecture: the **bottleneck**. The bottleneck is the narrowest point in the network (128×2×8×8 → flatten → 2048 → latent_dim) and directly determines the latent representation z that feeds the downstream classifier. If the bottleneck can produce a more discriminative representation, the classifier should benefit directly.

The champion's bottleneck is two sequential Conv3d layers with InstanceNorm and ReLU. This trial adds **SE attention** after the bottleneck convolutions, before spatial compression. The SE attention recalibrates the 128 channels of the bottleneck features, telling the network which channels carry diagnostically relevant information. This is applied at the most information-dense point in the network — right before the spatial dimensions are compressed to the latent vector.

Key prediction: SE attention at the bottleneck will produce a latent representation that is more discriminative for classification, because the attention operates on features that have already been processed by the successful encoder (ResSeparableConv3DBlock blocks). The encoder's successful training dynamics are preserved; only the bottleneck representation is enhanced.

I predict classification_accuracy_val will exceed the champion's 0.6083.

## Implementation
In `src/models/ae_models.py`:
- New model `AutoEncoder3D_AsymResSeparableBottleneckAttention`
- Encoder: identical to champion (ResSeparableConv3DBlock × 3 + SeparableConv3DBlock × 1)
- Bottleneck: Conv3d×2 (same as champion) + SEBlock3D(128, reduction=16) after convolutions
- Decoder: identical to champion (ResUpSeparableConv3DBlock × 3 + ResSeparableConv3DBlock)
- Same V4 asymmetric pooling: pool1=(1,2,2), z_pool3=(2,1,1)
- No skip connections

In `configs/autoencoder.yaml`:
- model_name: "AE3dAsymResSeparableBottleneckAttention"
- All HPs identical to champion: lr=5e-4, weight_decay=1e-5, dropout_rate=0.3, noise_std=0.0, patience=20

This isolates the bottleneck modification against the exact champion architecture.