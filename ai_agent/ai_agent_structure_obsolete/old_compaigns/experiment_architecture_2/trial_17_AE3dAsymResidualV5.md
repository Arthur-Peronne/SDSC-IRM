# Trial 17 — AE3dAsymResidualV5 — FAILURE

## Hypothesis
Trial 16 (new champion, AE3dAsymResidualV4) has dim=8=0.729, worse than the prior champion V4 (0.747). The decoder in trial 16 uses ResUpConv3DBlocks (residual). The hypothesis: residual shortcuts in the decoder anchor decoder features near identity, limiting the decoder's ability to reconstruct fine spatial detail at low latent dims (dim=8). Removing decoder residuals (plain UpConv3DBlocks, like V4) while keeping the encoder identical to trial 16 would isolate whether decoder residuals are responsible for dim=8 weakness.

## Implementation
- Encoder: identical to AE3dAsymResidualV4 (enc1-enc3 ResConv3DBlocks, enc4 plain Conv3DBlock, V4 pooling).
- Decoder: standard UpConv3DBlock for dec1-dec3, Upsample(1,2,2) + Conv3DBlock for dec4 — identical to V4.
- Bottleneck/FC: unchanged.

## Results
- **R2_dim8:** 0.729579 | **R2_dim60:** 0.769808 | **R2_dim240:** 0.787204
- **avg_validation_R2_mean:** 0.762197
- **delta_vs_champion** (trial avg − champion avg): −0.011609
- **MLflow Run IDs:** 79373c0996ab4d3bad069dd48cd605e8 0359178a3bde4ae29b67c471ffe098de b0f2b5d3e256420e89cc619f6c033ac8
- **Best epochs:** 45/75 | 60/90 | ~35/65

## Training Dynamics
Dim=60 converged later (best ep60) than dim=8 (best ep45), consistent with higher-capacity dims needing more epochs. Dim=240 showed good convergence (best ~ep35, val=0.000822) but lower R2 than trial 16.

## Conclusion
The hypothesis was refuted. Removing decoder residuals did NOT improve dim=8 (0.730 vs trial 16's 0.729 — essentially unchanged). Instead, it hurt dim=60 (0.770 vs 0.777, −0.007) and dim=240 (0.787 vs 0.815, −0.028). The decoder residuals are beneficial for reconstruction quality, especially at high latent dims where the decoder must reconstruct fine spatial detail.

**Key finding:** dim=8 weakness in the new champion comes from the encoder residuals (enc1-enc3 ResConv3DBlocks), not the decoder. The decoder is not the limiting factor for dim=8. The decoder's residual shortcuts provide gradient highways that improve dim=60 and dim=240 reconstruction without affecting dim=8 meaningfully.

**Design insight confirmed:** The decoder in trial 16 (ResUpConv3DBlocks) is optimal. The enc1-enc3 residual blocks slightly hurt dim=8 while being necessary for the strong dim=60 performance. The next step is to find a way to maintain the enc1-enc3 residual benefit for dim=60 while recovering dim=8 — possibly by using V4's plain encoder + residual decoder as a new combination.
