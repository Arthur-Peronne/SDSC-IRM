---
model_name: AE3dAsymResAttention
summary: Replace champion's ResSeparableConv3DBlock with ResAttentionConv3DBlock (standard conv + residual + SE attention) to combine gradient flow with channel-wise feature recalibration
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
Five consecutive trials (b606a10f, 0cadad28, 001b5a08, e4e99f58, 25eae42a) have failed by manipulating channel width (×2, ×0.5, encoder-only), weight_decay (100×), and dilated attention (1→2→4). The pattern is clear: capacity changes and aggressive dilation are not the levers that improve classification_accuracy_val.

This trial proposes a different combination: **standard convolutions + residual connections + SE attention**. The champion's success comes from residual connections (gradient flow, feature preservation) combined with separable convolutions (parameter efficiency). Trial 8 showed that attention is valuable but dilated convolutions were too aggressive.

The key insight: SE attention performs channel-wise feature recalibration, telling the network "which features matter." This is complementary to residual connections, which tell the network "what to preserve." Together, they should produce a latent representation that is both well-trained (via residuals) and selectively focused (via attention).

Unlike trial 8's dilated blocks (dilation 1→2→4, receptive field ~17×17×17), this model uses standard 3×3×3 convolutions (effective RF=7×7×7 with two layers), preserving the local spatial precision that the champion relies on. The SE attention adds the "what matters" signal without expanding the receptive field.

I predict classification_accuracy_val will exceed the champion's 0.6083.

## Implementation
In `src/models/ae_models.py`:
- New block `ResAttentionConv3DBlock`: standard 3x3x3 conv + residual shortcut + SE attention (reduction=16), MaxPool after attention
- New model `AutoEncoder3D_AsymResAttention`: all 4 encoder blocks use ResAttentionConv3DBlock
- Same V4 asymmetric pooling: pool1=(1,2,2), z_pool3=(2,1,1)
- Same bottleneck: Conv3d×2 → flatten → FC(2048, latent_dim)
- Decoder: identical to champion (ResUpSeparableConv3DBlock × 3 + ResSeparableConv3DBlock)
- No skip connections

In `configs/autoencoder.yaml`:
- model_name: "AE3dAsymResAttention"
- All HPs identical to champion: lr=5e-4, weight_decay=1e-5, dropout_rate=0.3, noise_std=0.0, patience=20

This isolates the encoder mechanism change (residual + standard conv + attention) against the exact champion HPs.