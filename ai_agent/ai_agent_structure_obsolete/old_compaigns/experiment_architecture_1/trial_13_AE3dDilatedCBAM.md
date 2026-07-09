# Trial 13 — AE3dDilatedCBAM — FAILURE

## Hypothesis
I will add CBAM-style spatial attention to the champion's encoder blocks, creating `AE3dDilatedCBAM`. The champion (`AE3dDilatedAttention`) already applies channel SE attention (which feature channels matter). Spatial attention adds a complementary mechanism: learning which 3D spatial locations matter. For cardiac MRI where the heart occupies a consistent but small region, spatial attention could help the encoder suppress background voxels and focus capacity on cardiac tissue, improving reconstruction fidelity of clinically relevant structures.

## Implementation
New `SpatialAttention3D` module: computes channel-wise average-pool and max-pool (each producing a 1×D×H×W map), concatenates them (2×D×H×W), applies Conv3d(2,1,k=7,p=3) + sigmoid to produce a spatial attention map, multiplies with the input feature map. New `CBAMDilatedConv3DBlock` wraps `DilatedConv3DBlock` + `SEBlock3D` (channel) + `SpatialAttention3D` (spatial), applied in that order. `AutoEncoder3D_DilatedCBAM` uses these blocks for enc1–enc4 with identical dilations (1,2,4,1), channel widths (1→8→16→32→64), bottleneck, FC, and decoder as the champion.

## Results
- **validation_R2_mean:** 0.771310
- **validation_R2_std:** 0.126245
- **val_R2_lower_bound** (mean − std): 0.645066
- **lower_bound_compared_to_champion** (trial lb − champion lb): -0.085669
- **mean_compared_to_champion** (trial mean − champion mean): -0.032328
- **MLflow Run ID:** 7cfefa33e5234c32895e2804b4eef5e8
- **Best epoch:** 50 / 80 (early stop)

## Training Dynamics
Early stopping at epoch 80 (best epoch 50). Convergence was reasonable but the validation loss plateaued around 0.000615 at best epoch. The validation std increased substantially relative to the champion (0.126 vs 0.073), indicating that spatial attention introduced higher patient-to-patient variance in reconstruction quality — the model's spatial focus is inconsistent across patients.

## Conclusion
The hypothesis failed. CBAM spatial attention degraded both mean R2 and lower bound vs the champion. Two mechanisms explain this:

1. **Spatial attention increases parameter count and variance.** Each encoder block gains a Conv3d(2,1,k=7) layer. With 150 patients and a high-dimensional spatial domain (32×128×128), these additional parameters overfit to training spatial patterns, producing the observed increase in validation std (0.126 vs champion's 0.073).

2. **Cardiac MRI may not benefit from explicit spatial gating at this scale.** The dilated convolutions in the champion already build a large receptive field, implicitly capturing global context including heart location. Adding an explicit spatial gate on top of this may create redundancy — the feature maps post-dilation already encode position implicitly, so the spatial attention module learns a noisy, redundant mask rather than a clean cardiac region gate.

The champion's channel SE alone is the right attention mechanism for this dataset size; spatial attention is an over-parameterization.
