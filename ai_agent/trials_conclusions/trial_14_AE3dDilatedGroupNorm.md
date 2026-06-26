# Trial 14 — AE3dDilatedGroupNorm — FAILURE

## Hypothesis
I will replace `InstanceNorm3d` with `GroupNorm(8 groups)` in all encoder blocks and bottleneck layers of the champion (`AE3dDilatedAttention`), creating `AE3dDilatedGroupNorm`. InstanceNorm normalizes each channel independently per sample, discarding cross-channel statistics. GroupNorm normalizes over groups of 8 channels, capturing intra-group feature co-activations. With channel counts (8, 16, 32, 64, 128) all divisible by 8, this is a clean drop-in. I predicted GroupNorm would reduce inter-patient variance by producing more consistent feature statistics across patients, potentially improving the lower bound.

## Implementation
New `DilatedConv3DGroupNormBlock`: identical to `DilatedConv3DBlock` but uses `nn.GroupNorm(num_groups, out_channels)` instead of `nn.InstanceNorm3d`. New `DilatedAttentionConv3DGroupNormBlock` wraps this with `SEBlock3D`. `AutoEncoder3D_DilatedGroupNorm` uses these blocks for enc1–enc4 (dilations 1,2,4,1; channels 1→8→16→32→64) and replaces `InstanceNorm3d` with `GroupNorm(8, 128)` in the bottleneck. Decoder uses the original `UpConv3DBlock` (InstanceNorm unchanged).

## Results
- **validation_R2_mean:** 0.732918
- **validation_R2_std:** 0.184706
- **val_R2_lower_bound** (mean − std): 0.548212
- **lower_bound_compared_to_champion** (trial lb − champion lb): -0.182523
- **mean_compared_to_champion** (trial mean − champion mean): -0.070720
- **MLflow Run ID:** ddd1462699c346bf8195c8338835040b
- **Best epoch:** 46 / 76 (early stop)

## Training Dynamics
Early stopping at epoch 76 (best epoch 46). Val loss at best epoch (0.000713) is worse than the champion's typical best. The validation std exploded to 0.185 — more than double the champion's 0.073 — indicating severe patient-to-patient inconsistency. Training R2 (0.855) versus validation R2 (0.733) shows a larger train/val gap than the champion, suggesting GroupNorm introduced mild overfitting.

## Conclusion
The hypothesis failed. GroupNorm degraded both mean R2 and, critically, the lower bound. The std increase (0.073 → 0.185) is the key signal — GroupNorm made the model *less* consistent across patients, the opposite of what was predicted.

Two mechanisms explain this:

1. **InstanceNorm provides stronger per-sample invariance for MRI.** Cardiac MRI intensities vary across patients due to scanner settings, contrast, and anatomy. InstanceNorm normalizes each sample's feature maps independently, acting as an implicit intensity normalization — each patient's features are placed on a common scale. GroupNorm normalizes over channel groups but retains cross-patient intensity variation within groups, making the model more sensitive to patient-specific signal magnitudes and increasing inter-patient variance.

2. **GroupNorm's group statistics are noisy at low channel counts.** The first encoder block has only 8 channels — with `num_groups=8`, each group contains exactly 1 channel, making GroupNorm equivalent to InstanceNorm there. At 16 channels, groups of 2 provide little statistical benefit. GroupNorm's advantage over InstanceNorm typically emerges at higher channel counts (≥32), which only applies to the later encoder stages. The normalization is therefore inconsistent across encoder stages.

InstanceNorm's per-sample normalization is the right choice for this dataset; GroupNorm does not confer a benefit here.
