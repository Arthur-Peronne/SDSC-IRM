# Trial 9 — AE3dDilatedAttentionWide — FAILURE

## Hypothesis
I will double all encoder channel widths in `AE3dDilatedAttention` (1→16→32→64→128 vs 1→8→16→32→64), creating `AE3dDilatedAttentionWide`, because wider channels give SE attention a richer feature bank at each spatial scale, allowing more diverse cardiac structural patterns to be represented simultaneously. The bottleneck is adjusted to start at 128 channels directly (removing the 64→128 expansion), keeping the flattened size at 2048 so FC layers remain unchanged. I predicted this would decrease `val_mse` by improving encoder expressiveness.

## Implementation
enc1–enc4 channel widths doubled: DilatedAttentionConv3DBlock(1,16), (16,32), (32,64), (64,128) with same dilations (1,2,4,1). bottleneck_conv adjusted to 128→128→128. All other components (final_down, FC layers, decoder) identical to champion.

## Results
- **val_mse:** 0.000848 (Δ +0.000275 vs champion 0.000573)
- **MLflow Run ID:** 7a41525da9e8444cbaf3ae2c45ae808a
- **Best epoch:** 35 / 65 (early stop)
- **validation_R2_mean:** 0.704 (degraded vs champion ~0.79)

## Training Dynamics
Convergence was slower than the champion — best epoch at 35 vs 63 for the champion, with early stopping at 65. The val loss stagnated around 0.000771 at the best epoch but MLflow val_mse is 0.000848 (computed on best restored model). Validation R2 dropped substantially (0.704 vs ~0.79), suggesting the wider model generalizes worse on this dataset.

## Conclusion
The hypothesis failed. Doubling channel widths increased the parameter count substantially without improving reconstruction quality. With only 150 patients (200 frames total), the wider model has more parameters to optimize but less data to constrain them, leading to a worse bias-variance trade-off. The champion's narrow channel progression (8→16→32→64) appears well-matched to the dataset size — it forces efficient feature compression at each scale rather than allowing redundant representations. The SE attention mechanism in the champion may already be effectively selecting the most informative channels from the narrow bank, so widening the bank provides diminishing returns while increasing optimization difficulty.
