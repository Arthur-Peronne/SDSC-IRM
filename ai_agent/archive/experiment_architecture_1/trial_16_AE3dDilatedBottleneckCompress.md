# Trial 16 — AE3dDilatedBottleneckCompress — FAILURE

## Hypothesis
I will add a 1×1×1 conv (128→64 channels) after `final_down` and before flattening, reducing the flattened feature vector from 2048 to 1024 before the FC projection. This halves the FC layer parameters and forces a learned channel selection step, acting as regularization that may benefit the small 150-patient dataset. The 1×1×1 conv learns which of the 128 spatial feature channels are most informative for reconstruction, allowing the FC to map from a pre-selected 1024-dim representation rather than a raw 2048-dim spatial unrolling.

## Implementation
New `AutoEncoder3D_DilatedBottleneckCompress`: identical to champion encoder and bottleneck, with an additional `compress` module after `final_down` — `Conv3d(128, 64, kernel_size=1) + InstanceNorm3d(64) + ReLU`. `feature_shape = (64, 1, 4, 4)`, `flattened_size = 1024`, `fc_enc = Linear(1024, 120)`. Decoder is unchanged: `fc_dec = Linear(120, 2048)`, reshape to `(128, 1, 4, 4)`, then 4 `UpConv3DBlock` as in the champion. The compression is encoder-only — the decoder still reconstructs from the full 2048-dim projection.

## Results
- **validation_R2_mean:** 0.769919
- **validation_R2_std:** 0.108136
- **val_R2_lower_bound** (mean − std): 0.661783
- **lower_bound_compared_to_champion** (trial lb − champion lb): -0.068952
- **mean_compared_to_champion** (trial mean − champion mean): -0.033719
- **MLflow Run ID:** 9764a649c62f40e8afcff830705ec28f
- **Best epoch:** 41 / 71 (early stop)

## Training Dynamics
Early stopping at epoch 71 (best epoch 41) — earlier convergence than the champion, suggesting the compressed bottleneck limits the useful learning signal. Validation std increased to 0.108 (vs champion 0.073), indicating more inter-patient variance — the opposite of the intended regularization effect.

## Conclusion
The hypothesis failed. The 1×1×1 compression degraded both mean R2 and variance. Two mechanisms explain this:

1. **Information loss at the compression step.** The encoder encodes spatial features into 128 channels across a (1,4,4) spatial map. Compressing to 64 channels with a 1×1×1 conv discards half the channel capacity before the latent projection. With latent_dim=120, the information pathway is already highly compressed (128×1×4×4=2048 → 120). Adding a 128→64 step creates an additional bottleneck that forces redundant information loss, reducing reconstruction quality.

2. **Asymmetric encoder-decoder bottleneck.** The decoder's `fc_dec` still maps to 2048 features (128×1×4×4), meaning the decoder has more representational freedom than the encoder can provide. The encoder's compressed 1024-dim intermediate cannot efficiently encode the spatial structure the decoder is trying to reconstruct from 2048 targets. This asymmetry impairs the encoder-decoder alignment during training.

The champion's direct flatten from (128,1,4,4) to 2048 is the right approach — the spatial feature volume is already compact enough that further channel compression is harmful rather than helpful.
