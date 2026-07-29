# Trial 23 — AE3dDilatedDecoder — FAILURE

## Hypothesis
The champion encoder uses multi-scale dilations (1,2,4,1) that progressively expand and then consolidate the receptive field. The decoder uses plain UpConv3DBlock (dilation=1 everywhere). Hypothesis: mirroring the encoder's multi-scale approach in the decoder — using anisotropic dilation (depth=1, dilation_hw=4 at dec1; depth=1, dilation_hw=2 at dec2) — should allow the decoder to use wide spatial context when reconstructing global cardiac structure from compressed features. Zero-parameter change (same 3×3×3 kernel weight count, different receptive field).

## Implementation
New `DilatedUpConv3DBlock(in_ch, out_ch, dilation_hw)`: ConvTranspose3d (upsampling) + 2×Conv3d(k=3, dilation=(1,dh,dh), padding=(1,dh,dh), IN+ReLU). Champion encoder and bottleneck unchanged. dec1: DilatedUpConv3DBlock(128,64, dilation_hw=4), dec2: DilatedUpConv3DBlock(64,32, dilation_hw=2), dec3/dec4: plain UpConv3DBlock. Total params: 2,021,857 (identical to champion).

## Results
- **validation_R2_mean:** 0.745067
- **validation_R2_std:** 0.127173
- **val_R2_lower_bound** (mean − std): 0.617894
- **lower_bound_compared_to_champion** (trial lb − champion lb): -0.112841
- **mean_compared_to_champion** (trial mean − champion mean): -0.058571
- **MLflow Run ID:** a81485a401b6489eabcb624bc62c1cf5
- **Best epoch:** 34 / 64 (early stop)

## Training Dynamics
Very early stopping at epoch 64 (best epoch 34) — the second-earliest stopping in Phase 2, indicating significant convergence difficulty. Train R2=0.814 (the lowest training R2 in Phase 2), confirming the dilated decoder is harder to optimize. Std=0.127.

## Conclusion
The hypothesis failed. Wide-receptive-field decoder (dilation_hw=4 at dec1) impairs reconstruction rather than improving it. Two mechanisms:

1. **Reconstruction benefits from locality, not globality.** The encoder's multi-scale dilations are designed to *capture* features at different spatial scales — encoding requires seeing both local texture and global context. The decoder's task is *reconstruction*: given global context from the latent space (already encoded globally by the bottleneck FC), the decoder must place specific features at specific spatial locations. Dilated decoder convolutions at dec1 look at spatially dispersed locations when trying to fill in local spatial structure — this creates spatial confusion during reconstruction.

2. **Dilation_hw=4 at dec1 has near-full spatial coverage.** After initial_up, the feature map is 128×2×8×8. With dilation_hw=4 and kernel_size=3, the effective H/W span is 9×9, covering the entire 8×8 feature map. Each output location's receptive field overlaps with every other location. This destroys spatial locality in the first decoder block, making it impossible to learn position-specific reconstruction patterns.

3. **Training is destabilized.** Best epoch 34 vs champion's ~70 epochs — the dilated decoder converges to a poor local minimum quickly and then fails to improve, suggesting the optimization landscape is fundamentally harder with dilated decoder blocks.

**Confirmed:** Anisotropic dilation in the decoder is counter-productive. The plain UpConv3DBlock decoder is well-suited for local reconstruction at each scale.
