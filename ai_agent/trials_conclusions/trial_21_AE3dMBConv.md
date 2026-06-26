# Trial 21 — AE3dMBConv — FAILURE

## Hypothesis
The champion's encoder (~220K parameters across 4 DilatedAttentionConv blocks) may be overparameterized for 100 training patients. An inverted bottleneck (MBConv-style) encoder replaces each standard conv block with: (1) pointwise expansion (1×1×1 conv, ×4 channels), (2) depthwise spatial conv (3×3×3, per-channel, with dilation), (3) pointwise projection (1×1×1 conv, back to target channels). This reduces encoder parameters from ~220K to ~23K (10× fewer) while preserving dilations (1,2,4,1) and keeping the champion's bottleneck and decoder unchanged. The hypothesis: a dramatically lighter encoder cannot memorize patient-specific spatial patterns, forcing it to learn more generalizable features, reducing inter-patient variance.

## Implementation
New `MBConv3DBlock(in_ch, out_ch, expand_ratio=4, dilation, downsample)`: expand (1×1×1, in_ch→in_ch×4, IN+ReLU) → depthwise (3×3×3 groups=hidden, dilation, IN+ReLU) → project (1×1×1, hidden→out_ch, IN+ReLU) → optional MaxPool. New `AutoEncoder3D_MBConv`: enc1-enc4 using MBConv3DBlock with dilations (1,2,4,1); champion bottleneck_conv, final_down, FC, and decoder unchanged. Total params: 1,824,169 (-197,688 vs champion).

## Results
- **validation_R2_mean:** 0.739262
- **validation_R2_std:** 0.179705
- **val_R2_lower_bound** (mean − std): 0.559557
- **lower_bound_compared_to_champion** (trial lb − champion lb): -0.171178
- **mean_compared_to_champion** (trial mean − champion mean): -0.064376
- **MLflow Run ID:** 5a62d08cac024e72aaaf020269338e72
- **Best epoch:** 60 / 90 (early stop)

## Training Dynamics
Early stopping at epoch 90 (best epoch 60). Training R2=0.917 vs validation R2=0.739 — the largest train/val gap in Phase 2 despite having fewer total parameters. Std=0.180, worst of all except trial 20 (0.188).

## Conclusion
The hypothesis failed. Counterintuitively, the lighter encoder produces more overfitting (train/val gap = 0.178) than the champion (train/val gap ≈ 0.085). Two mechanisms explain this:

1. **Overfitting is concentrated in the bottleneck/decoder, not the encoder.** The champion's 1.8M parameters are in the bottleneck FC layers (2048↔120) and the decoder conv blocks. These account for the vast majority of both the champion's and trial 21's parameters. Cutting encoder parameters by 10× does not meaningfully reduce the total model capacity available for memorization. The bottleneck FC (2048×120 = 245,760 params) alone exceeds the entire trial 21 encoder (22,896 params) by 10×.

2. **Weaker encoder features force the decoder to do more work.** The MBConv encoder, with its depthwise spatial convolutions, cannot perform cross-channel mixing in the spatial filtering step. This produces weaker, less discriminative feature maps at enc4 (before the bottleneck). Weaker feature maps at the bottleneck input force the large decoder to "hallucinate" more structure during reconstruction, rather than decode from richly-structured features. This hallucination process memorizes training-specific reconstruction patterns.

3. **The inverted bottleneck is designed for classification, not reconstruction.** MBConv blocks optimize for semantic features (image classification) by discarding spatial detail in the expand→depthwise→project chain. For reconstruction, spatial detail preservation is critical. The champion's full 3×3×3 convolutions with channel mixing precisely preserve and transform spatial structure that the decoder needs to reconstruct the MRI.

**Key finding:** The overfitting problem in this 150-patient dataset is not localized to the encoder. The large bottleneck FC layer (2048-dim flatten + 120-dim projection) and the decoder (~1.7M params) are the primary sources of overfitting. Reducing encoder capacity alone is ineffective.
