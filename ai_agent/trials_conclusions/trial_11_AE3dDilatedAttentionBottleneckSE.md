# Trial 11 — AE3dDilatedAttentionBottleneckSE — FAILURE

## Hypothesis
I will add SE channel attention inside the bottleneck of `AE3dDilatedAttention`, creating `AE3dDilatedAttentionBottleneckSE`. The champion's bottleneck contains two sequential Conv3d→InstanceNorm3d→ReLU blocks that transform the spatial representation from 64 to 128 channels before flattening to the latent vector. By inserting an `SEBlock3D(128, reduction=16)` after each of these two blocks, the model can recalibrate which of the 128 bottleneck channels are most informative for the cardiac MRI reconstruction. The mechanism: SE attention learns global channel statistics at the bottleneck and suppresses redundant channels, producing a cleaner 128-channel representation before the FC projection to 120 latent dimensions. I predicted this would decrease `val_mse` by improving the quality of information entering the latent space.

## Implementation
`AutoEncoder3D_DilatedAttentionBottleneckSE` mirrors the champion exactly except the bottleneck:
- `bottleneck_conv1` → `bottleneck_se1` (SEBlock3D, 128ch, reduction=16) → `bottleneck_conv2` → `bottleneck_se2` (SEBlock3D, 128ch, reduction=16) → `final_down` → flatten → `fc_enc`
- Encoder (enc1–enc4), decoder, and FC layers are identical to the champion.
- Verified with forward pass: output [1,1,32,128,128], latent [1,120].

## Results
- **val_mse:** 0.000661 (Δ +0.000087 vs champion 0.000573)
- **MLflow Run ID:** a395c3ef1038475eb067625a324e6e08
- **Best epoch:** 46 / 76 (early stop)
- **validation_R2_mean:** 0.790

## Training Dynamics
Training val loss at best epoch was 0.000595 (online), but the MLflow val_mse computed on the restored best model is 0.000661. LR decayed from 5e-5 to 2.5e-5 around epoch 45, triggering the best epoch at 46; subsequent LR halving did not unlock further improvement. Early stopping at epoch 76 (30 epochs after best epoch 46). The convergence pattern is similar to Trial 10 (residual) — reasonable convergence speed but final quality below the champion.

## Conclusion
The hypothesis failed. Adding SE attention to the bottleneck did not improve upon the champion's simple two-conv bottleneck. Several reasons may explain this:

1. **SE at bottleneck is redundant with enc4**: The final encoder block (`enc4`) already applies `DilatedAttentionConv3DBlock` which includes SE attention at 64 channels before the bottleneck. The bottleneck SE may be recalibrating channels that are already partially recalibrated, yielding diminishing returns.

2. **Bottleneck SE adds parameters without adding spatial structure**: SE at 128 channels (reduction=16) adds 2×(128/16×128 + 128×128/16) ≈ 4096 extra parameters. These parameters compete for information from a fixed 150-patient dataset, potentially worsening the bias-variance trade-off slightly.

3. **The champion's bottleneck may already be optimal**: Four consecutive failures (Trials 8–11) against the champion across depth, width, residual shortcuts, and now bottleneck SE all suggest that `AE3dDilatedAttention` is at or near a local optimum for this dataset size. The champion's design — narrow channel progression (8→16→32→64) with dilated+SE attention at each encoder scale — appears already well-matched to the expressiveness requirements of 150 cardiac MRI patients.

**Future directions**: Consider exploring the decoder (currently symmetric upconv blocks without attention), or training configuration changes (data augmentation, different split sizes), or a fundamentally different approach (e.g., asymmetric encoder-decoder capacity, VAE-style regularization of the latent space).
