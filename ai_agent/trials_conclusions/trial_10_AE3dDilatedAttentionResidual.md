# Trial 10 — AE3dDilatedAttentionResidual — FAILURE

## Hypothesis
I will add intra-block residual shortcuts to each of the 4 encoder stages of `AE3dDilatedAttention`, creating `AE3dDilatedAttentionResidual`. Each block computes: output = ReLU(SE(DilatedConv(x)) + proj(x)), followed by MaxPool. The projection is a 1×1×1 conv when in_channels ≠ out_channels. This is not a skip connection between encoder and decoder — it is a local shortcut within each block. The mechanism: residual paths provide direct gradient highways through the 4-block encoder stack, making it easier for each block to learn to refine features (learn the residual) rather than transform them from scratch. I predicted this would decrease `val_mse` by improving optimization stability of the stacked encoder.

## Implementation
New block class `ResidualDilatedAttentionConv3DBlock`: DilatedConv3DBlock (downsample=False) → SEBlock3D → ReLU(out + shortcut) → optional MaxPool. Shortcut uses nn.Conv3d(in,out,1) when in≠out, else nn.Identity(). `AutoEncoder3D_DilatedAttentionResidual` uses this block for enc1–enc4 with same channel widths (1→8→16→32→64) and dilations (1,2,4,1) as the champion. Bottleneck and decoder identical to champion.

## Results
- **val_mse:** 0.000643 (Δ +0.000070 vs champion 0.000573)
- **MLflow Run ID:** c7fb0c44e9ff454580442634bff8da98
- **Best epoch:** 49 / 79 (early stop)
- **validation_R2_mean:** 0.778

## Training Dynamics
Stable convergence with no instability. Steady improvement through epoch 49, then plateauing. This is the closest result to the champion among Trials 8–10 (Δ +0.000070 vs +0.000103 for Trial 8 and +0.000275 for Trial 9), suggesting residual connections do provide some benefit over pure capacity increases. LR decay at epoch 53 did not unlock further improvement.

## Conclusion
The hypothesis partially held — residual connections did improve gradient flow, producing the best result of the current session's failures and converging more stably than the non-residual variants. However, the shortcut paths may interfere with the champion's compression objective: by providing an easy bypass for each block, the residual connection reduces the pressure on each stage to produce a maximally informative compressed representation. The champion's blocks, without shortcuts, are forced to learn efficient dilated+attention representations at each scale; the residual variant can "coast" on the identity path. Additionally, the SE attention operates on the non-residual path only (before adding the shortcut), so it cannot recalibrate the combined representation. An alternative design where SE operates after the residual sum might perform differently.
