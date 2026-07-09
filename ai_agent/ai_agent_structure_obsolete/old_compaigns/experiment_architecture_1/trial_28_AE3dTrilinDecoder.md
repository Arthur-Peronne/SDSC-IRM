# Trial 28 — AE3dTrilinDecoder — FAILURE

## Hypothesis
Cooldown triggered by trials 25-27 (3 consecutive AE3dDilatedAttention FAILURE trials). Trial 28 must be Exploration. ConvTranspose3d is known to produce checkerboard artifacts — spatial aliasing patterns from the stride-2 transposed convolution. These artifacts can vary in severity across patients with different cardiac geometries, contributing to inter-patient reconstruction variance. Hypothesis: replacing ConvTranspose3d with parameter-free trilinear interpolation + Conv3d separates the upsampling step (smooth, artifact-free) from the channel reduction step (learned Conv3d), potentially reducing patient-specific upsampling artifacts.

## Implementation
`AutoEncoder3D_TrilinDecoder`: champion encoder, bottleneck, and FC layers unchanged. New `TrilinUpConv3DBlock(in_channels, out_channels)`: F.interpolate(scale_factor=2, mode='trilinear', align_corners=False) + Conv3d(in,out,k=3,p=1)+IN+ReLU + Conv3d(out,out,k=3,p=1)+IN+ReLU. initial_up replaced by inline F.interpolate. Total params: 1,950,377 (−71,480 vs champion).

## Results
- **validation_R2_mean:** 0.763412
- **validation_R2_std:** 0.125589
- **val_R2_lower_bound** (mean − std): 0.637823
- **lower_bound_compared_to_champion** (trial lb − champion lb): -0.092912
- **mean_compared_to_champion** (trial mean − champion mean): -0.040226
- **MLflow Run ID:** e60eeeec9b3e4682bea36fb11a0daef4
- **Best epoch:** 46 / 76 (early stop)

## Training Dynamics
Early stopping at epoch 76 (best epoch 46). Std=0.126, nearly identical to the trilinear decoder mean (0.763 vs 0.734 for comparable failures). Train R2=0.849 vs val R2=0.763 — train/val gap persists.

## Conclusion
The hypothesis failed. Replacing ConvTranspose3d with trilinear interpolation does not reduce inter-patient reconstruction variance. This confirms that ConvTranspose3d checkerboard artifacts are NOT a significant contributor to the champion's variance advantage. The champion's low std=0.073 comes from its specific combination of dilated attention encoder, two-layer bottleneck, and plain UpConv3DBlock decoder — not from the absence of checkerboard artifacts.

Key insight: the trilinear decoder has different capacity distribution — the first Conv3d after interpolation takes in_channels (128, 64, 32, 16) rather than the champion's ConvTranspose3d output channels (64, 32, 16, 8). This means the channel reduction happens at full upsampled resolution, which requires more parameters in that step (Conv3d(128,64,k3)=221K vs ConvTranspose3d(128,64,k2)=65K) but fewer overall parameters (due to removing the initial_up ConvTranspose3d). The different capacity distribution doesn't improve generalization.

**Lesson:** The decoder upsampling mechanism (ConvTranspose3d vs trilinear) doesn't drive variance. The champion's decoder is already near-optimal in its current form. Only channel attention (SE r=4, trial 15) provides any beneficial modification to the decoder.
