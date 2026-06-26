# Trial 24 — AE3dDilatedDecRefinement — FAILURE

## Hypothesis
Trial 23 confirmed the decoder needs locality, not globality. Trial 22 showed capacity reduction doesn't help. The only successful decoder modification was trial 15 (SE all 4 blocks, Δlb=-0.005). Hypothesis: adding a final non-upsampling refinement block at full resolution (32×128×128) after dec4 but before final_conv gives the decoder one extra "polishing" stage at the finest scale, potentially improving reconstruction of fine cardiac texture. The block is Conv3DBlock(8,8,ds=False): two Conv3d(8,8,k=3,p=1)+IN+ReLU layers, +3,472 params.

## Implementation
`AutoEncoder3D_DilatedDecRefinement`: champion encoder (DilatedAttentionConv3DBlock enc1-enc4, dilations 1/2/4/1, reduction=16) and bottleneck (Conv3d 64→128→128, final_down) unchanged. Champion decoder (UpConv3DBlock dec1–dec4) unchanged. After dec4: `self.refine = Conv3DBlock(8, 8, downsample=False)`. final_conv: Conv3d(8,1,k=3,p=1). Total params: 2,025,329 (+3,472 vs champion).

## Results
- **validation_R2_mean:** 0.733552
- **validation_R2_std:** 0.120104
- **val_R2_lower_bound** (mean − std): 0.613448
- **lower_bound_compared_to_champion** (trial lb − champion lb): -0.117287
- **mean_compared_to_champion** (trial mean − champion mean): -0.070086
- **MLflow Run ID:** e2eeb61dc9c24e7089cbae9e31e2f51c
- **Best epoch:** 36 / 66 (early stop)

## Training Dynamics
Early stopping at epoch 66 (best epoch 36). Train R2=0.837 vs val R2=0.734 — clear train/val gap. std=0.120, more than 1.6× the champion's 0.073. The refinement block failed to improve reconstruction quality and increased variance substantially.

## Conclusion
The hypothesis failed. Adding a full-resolution Conv3DBlock after dec4 does not improve generalization. Two mechanisms:

1. **Extra capacity at full resolution enables overfitting.** At 32×128×128 (524,288 voxels) with 100 training patients, a non-upsampling block has the most parameters per output voxel in the network. This creates a strong overfitting risk: the block can memorize patient-specific texture patterns rather than learning generalizable reconstruction priors.

2. **The final_conv is already the right refinement stage.** Champion's final_conv (Conv3d(8,1,k=3,p=1)) produces the 1-channel output from 8 feature channels. Adding another 8→8 block before it doesn't improve the mapping quality — it creates an unnecessary intermediate representation that the network must compress back through the same final_conv bottleneck.

**Cumulative decoder lesson:** The champion decoder is near-optimal in its current form. The only beneficial decoder modification found across all trials is SE channel calibration on all 4 decoder blocks (trial 15, Δlb=-0.005). Everything else — adding blocks (trial 24), dilation (trial 23), partial SE (trial 19) — increases std and reduces lb. The decoder does not benefit from additional capacity or complexity.
