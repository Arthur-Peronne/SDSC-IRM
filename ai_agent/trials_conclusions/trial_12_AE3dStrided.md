# Trial 12 — AE3dStrided — FAILURE

## Hypothesis
I will replace MaxPool3d downsampling with learned stride-2 Conv3d+InstanceNorm3d+ReLU in `AE3dDilatedAttention`, creating `AE3dStrided`. MaxPool discards spatial information by hard-selecting local maxima; a learned stride-2 convolution can optimize the spatial aggregation function for the specific features produced by each encoder stage. All other components (dilated convolutions, SE attention, same channel widths 1→8→16→32→64, bottleneck, FC, decoder) are identical to the champion. I predicted this would improve reconstruction by preserving more task-relevant spatial structure through the downsampling steps.

## Implementation
Two new block classes: `StridedDilatedConv3DBlock` (DilatedConv3DBlock with `stride_down = Conv3d(out,out,3,stride=2,padding=1)+IN+ReLU` replacing MaxPool) and `StridedDilatedAttentionConv3DBlock` (wraps the strided block with SEBlock3D). `AutoEncoder3D_Strided` uses these for enc1–enc4 with identical dilations (1,2,4,1) and channel widths. Bottleneck, final_down, FC layers, and decoder are copied exactly from the champion. Forward pass verified: output (1,1,32,128,128), latent (1,120).

## Results
- **validation_R2_mean:** 0.675142
- **validation_R2_std:** 0.167746
- **val_R2_lower_bound** (mean − std): 0.507395
- **lower_bound_compared_to_champion** (trial lb − champion lb): -0.223340
- **mean_compared_to_champion** (trial mean − champion mean): -0.128496
- **MLflow Run ID:** 1d85705532664fba893f72588a9ea0aa
- **Best epoch:** 48 / 78 (early stop)

## Training Dynamics
Rapid initial convergence (best epoch at 48), but the validation loss plateaued around 0.000910 at best epoch and remained there through LR decay steps until early stopping at epoch 78. The train loss continued decreasing steadily (0.000512 at epoch 78 vs 0.000910 val at best epoch), suggesting a train/val gap — the stride-2 conv layers overfit the training set slightly. The high val_std (0.168) relative to the champion (0.073) confirms poor generalization.

## Conclusion
The hypothesis failed. Replacing MaxPool with stride-2 convolutions significantly degraded performance (R2_mean drops from 0.804 to 0.675, lower bound drops by 0.223). Two mechanisms explain this:

1. **MaxPool as implicit regularization:** MaxPool's non-parametric nature prevents overfitting in the downsampling step. The stride-2 conv adds learnable parameters at each downsampling stage — with only 150 patients, these extra parameters worsen the bias-variance trade-off, producing the observed train/val gap.

2. **MaxPool's translation invariance may be beneficial:** For cardiac MRI where the heart position slightly varies between patients, MaxPool provides local translation invariance at each scale, which may help the encoder produce more consistent features across patients. The learned stride-2 conv lacks this inductive bias.

The champion's use of MaxPool is not a weakness to exploit but a structural strength for this dataset size and domain.
