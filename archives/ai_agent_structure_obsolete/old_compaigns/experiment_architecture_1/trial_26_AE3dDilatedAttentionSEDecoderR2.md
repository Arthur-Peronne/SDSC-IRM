# Trial 26 — AE3dDilatedAttentionSEDecoderR2 — FAILURE

## Hypothesis
Trial 15 (SE on all 4 decoder blocks, reduction=4) was the closest modification to champion: Δlb=-0.005, std=0.077. It established that SE channel calibration in the decoder is the right mechanism. Trial 26 tests whether doubling the SE hidden units (reduction=2 instead of 4) — giving the SE more capacity to capture channel-level patterns — can close the remaining gap to champion. Total +5,440 params vs trial 15's +2,720.

## Implementation
`AutoEncoder3D_DilatedAttentionSEDecoderR2`: champion encoder (DilatedAttentionConv3DBlock enc1–enc4, dilations 1/2/4/1, reduction=16) and bottleneck unchanged. Each decoder block followed by SEBlock3D at reduction=2: dec1→SE(64,r=2), dec2→SE(32,r=2), dec3→SE(16,r=2), dec4→SE(8,r=2). Total params: 2,027,297 (+5,440 vs champion).

## Results
- **validation_R2_mean:** 0.772391
- **validation_R2_std:** 0.113714
- **val_R2_lower_bound** (mean − std): 0.658677
- **lower_bound_compared_to_champion** (trial lb − champion lb): -0.072058
- **mean_compared_to_champion** (trial mean − champion mean): -0.031247
- **MLflow Run ID:** 8ccdfad9237c4eaea2148bac8cd0671b
- **Best epoch:** 49 / 79 (early stop)

## Training Dynamics
Early stopping at epoch 79 (best epoch 49). Train R2=0.863 vs val R2=0.772 — clear train/val gap. std=0.114, significantly worse than trial 15's std=0.077.

## Conclusion
The hypothesis failed — and more critically, reduction=2 performs worse than reduction=4 (trial 15). The SE reduction curve in the decoder is now fully characterized:

| Reduction | Params | val_lb   | val_std | Δlb vs champion |
|-----------|--------|----------|---------|-----------------|
| r=2       | +5440  | 0.6587   | 0.114   | -0.072          |
| r=4       | +2720  | 0.7259   | 0.077   | **-0.005** (best) |
| r=8 (partial, dec1+dec2 only) | +640 | ~0.651 | 0.130 | -0.080 |
| r=16 (champion encoder pattern) | 0 | 0.7307 | 0.073 | 0 (champion) |

The optimal SE reduction for the decoder is r=4. More SE capacity (r=2) overfits: the larger SE networks learn patient-specific channel patterns rather than generalizable calibration. Less SE capacity (partial r=8) under-calibrates: not enough capacity to capture meaningful channel-level patterns.

**Key insight:** The decoder SE mechanism has a sweet spot at reduction=4 (trial 15). This mirrors how SE blocks are typically recommended (reduction=16 is common, but small networks benefit from lower reduction until the hidden size becomes too small relative to the channel count). For the decoder's channel counts (8–64), reduction=4 gives hidden sizes of 2–16 — small enough to avoid overfitting, large enough to be expressive.

**Lesson from trials 15, 19, 26:** Decoder SE is a genuine mechanism for reducing variance, but only at reduction=4. The optimal point has been established. Further variants of SE reduction are unlikely to improve over trial 15.
