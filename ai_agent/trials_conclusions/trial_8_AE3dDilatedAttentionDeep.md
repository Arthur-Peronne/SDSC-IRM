# Trial 8 — AE3dDilatedAttentionDeep — FAILURE

## Hypothesis
I will add a 5th non-downsampling DilatedAttention encoder block (64→64, dilation=2) between enc4 and the bottleneck of `AE3dDilatedAttention` because at the coarsest spatial resolution (2×8×8), an extra SE-attention pass should suppress non-cardiac channels before bottleneck expansion to 128, extending the effective receptive field without spatial resolution loss. I predicted this would decrease `val_mse` by giving the bottleneck cleaner, more focused feature representations.

## Implementation
Added `self.enc5 = DilatedAttentionConv3DBlock(64, 64, dilation=2, downsample=False, reduction=16)` after enc4. Called in `encode()` between enc4 and bottleneck_conv. All other components (bottleneck, FC layers, decoder) are identical to the champion. Spatial flow: input → enc1–4 (each ×2 downsample) → enc5 (no downsample, dilation=2) → bottleneck_conv (64→128→128) → final_down → flatten → fc_enc.

## Results
- **val_mse:** 0.000677 (Δ +0.000103 vs champion 0.000573)
- **MLflow Run ID:** 7c5c45e204ef4bea9c0eddded525fc9a
- **Best epoch:** 45 / 75 (early stop)
- **validation_R2_mean:** 0.791

## Training Dynamics
Smooth convergence with no instability or spikes. The loss curve descended cleanly and early stopping triggered at epoch 75. LR decay kicked in at epoch 52, producing further refinement but no breakthrough. Convergence speed similar to the champion (early stop at epoch 75 vs 63 for champion).

## Conclusion
The hypothesis did not hold. Adding a 5th encoder block at the coarsest spatial resolution (2×8×8) does not improve reconstruction. At this stage the spatial dimensions are already heavily compressed (2×8×8 = 128 voxels), and additional 3×3 dilated convolutions with dilation=2 operate on a scale that exceeds the available spatial extent in the depth dimension (only 2 slices). The SE channel attention may not gain useful signal when feature maps are this small — there is insufficient spatial context for meaningful squeeze operations. The capacity increase did not translate into better latent representations; instead it may have introduced slight overfitting at the bottleneck stage. The champion's bottleneck design (direct 64→128 expansion followed by stride-2 final_down) appears well-calibrated for this input resolution.
