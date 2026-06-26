# Trial 22 — AE3dCompact — FAILURE

## Hypothesis
Trial 21 revealed that a lighter encoder (22K params) still produces high validation variance (std=0.180), because the bottleneck (~1.3M params) and decoder (~500K params) dominate total model capacity. Hypothesis: reducing the bottleneck from 128ch to 64ch (no expansion) and halving decoder channels throughout (128→64→32→16→8 becomes 64→32→16→8→4) should reduce the capacity available for overfitting in the bottleneck/decoder path, while the champion encoder (unchanged) continues to extract strong features. Total params: 850K (-58% vs champion).

## Implementation
`AutoEncoder3D_Compact`: champion encoder (DilatedAttentionConv3DBlock enc1-enc4, reduction=16) unchanged. Bottleneck: Conv3d(64,64)×2 instead of Conv3d(64,128)+Conv3d(128,128). final_down: Conv3d(64,64,k=2,s=2). feature_shape=(64,1,4,4), flattened=1024. FC enc/dec: 1024↔120 instead of 2048↔120. Decoder: UpConv(64,32), UpConv(32,16), UpConv(16,8), UpConv(8,4), final_conv(4,1).

## Results
- **validation_R2_mean:** 0.748827
- **validation_R2_std:** 0.158273
- **val_R2_lower_bound** (mean − std): 0.590554
- **lower_bound_compared_to_champion** (trial lb − champion lb): -0.140181
- **mean_compared_to_champion** (trial mean − champion mean): -0.054811
- **MLflow Run ID:** 34c8dc9c2a2c4cc1925caa009d6d88d8
- **Best epoch:** 61 / 91 (early stop)

## Training Dynamics
Early stopping at epoch 91 (best epoch 61). Std=0.158, still more than twice the champion's 0.073. Train R2=0.854 vs val R2=0.749 — clear train/val gap despite 58% parameter reduction.

## Conclusion
The hypothesis failed. Reducing model capacity from 2M to 850K parameters did not reduce inter-patient variance. This establishes a critical finding: **the champion's low variance (std=0.073) is not achievable through capacity reduction alone.**

Three key observations from trials 19-22:
1. **Lighter encoder (trial 21, 22K encoder params) → std=0.180.** Weaker encoder features force decoder memorization.
2. **Lighter decoder (trial 22, 850K total) → std=0.158.** Reduced decoder capacity cannot reconstruct diverse cardiac patterns, increasing patient-specific variance.
3. **The champion's parameter count is load-bearing** — it has approximately the right number of parameters for this task. Too few parameters (under-capacity) causes reconstruction failure and variance; too many (added parameters in most other trials) causes overfitting. The champion sits in a narrow sweet spot.

The one modification that achieved close-to-champion variance was trial 15 (SE decoder, std=0.077), which added only 2,720 parameters (0.13% increase). This was a qualitative improvement to the decoder's generalization mechanism, not a capacity reduction. The lesson: to reduce variance, improve the **mechanism** (SE channel calibration), not the **quantity** (parameter count).
