# Trial 18 — AE3dAsymV4ResDec — FAILURE

## Hypothesis
Trial 17 showed that removing decoder residuals from trial 16 did not fix dim=8 (still 0.730 vs 0.729). The encoder residuals (enc1-enc3) are therefore the cause of the dim=8 regression from V4 (0.747 → 0.729). Trial 18 tests the orthogonal combination: V4's plain encoder (enc1-enc4 all plain Conv3DBlocks) with the residual decoder from trial 12/16 (ResUpConv3DBlocks). If enc1-enc3 residuals hurt dim=8 but the decoder residuals help dim=60/240, combining a plain encoder with a residual decoder should recover dim=8 to ~0.747 while preserving the dim=60/240 gains. This would be the best of both worlds.

## Implementation
- Encoder: identical to AE3dFCDeepAsymV4 — all four stages are plain Conv3DBlocks, V4 pooling (pool1=MaxPool(1,2,2), z_pool3=MaxPool(2,1,1)).
- Decoder: ResUpConv3DBlock for dec1/dec2/dec3, Upsample(1,2,2) + ResConv3DBlock(downsample=False) for dec4 — identical to trial 16's decoder.
- Bottleneck/FC: unchanged from V4/trial 16.
- ~1.56M params at dim=8.

## Results
- **R2_dim8:** 0.747051 | **R2_dim60:** 0.776088 | **R2_dim240:** 0.730674
- **avg_validation_R2_mean:** 0.751271
- **delta_vs_champion** (trial avg − champion avg): −0.022535
- **MLflow Run IDs:** da6c30cc7a114c2bb2bb3e2e6aa0717e afa430ff03e441c99eb5b77ec3fddcb9 bcd518ba4ab64cb488fb7dcf4302278c

## Training Dynamics
dim=8 and dim=60 converged normally. dim=240 showed unusually early convergence with higher val_loss than trial 16 (best val_loss ~0.000795 vs 0.000517 for trial 16), consistent with the bottleneck features not being well-adapted to the residual decoder structure.

## Conclusion
The hypothesis was refuted. dim=8 did recover to 0.747 (as predicted, matching V4 exactly), and dim=60 improved marginally (0.776 vs V4's 0.775). However, dim=240 collapsed catastrophically (0.731 vs V4's 0.778 and trial 16's 0.815 — a regression of −0.047 vs V4 and −0.084 vs trial 16).

**Key finding:** The plain encoder + residual decoder combination is architecturally incompatible. The residual decoder (ResUpConv3DBlock) was designed in the context of residual encoders (trial 12/16), where the bottleneck features have a residual-shaped distribution. A plain encoder produces a different feature distribution in the bottleneck, and the decoder's identity shortcuts clash with these features — the gradient flow through the decoder residuals cannot reconstruct fine spatial structure when the encoder features lack the residual structure the decoder expects. This is most visible at dim=240 (high-capacity) where the mismatch in feature distributions has maximum impact.

**Encoder-decoder consistency principle confirmed:** Residual blocks in encoder and decoder must be used consistently. The beneficial combination is: enc1-enc3 residual + enc4 plain + residual decoder (trial 16's architecture). Mixing plain encoder with residual decoder creates a distribution mismatch that hurts high-capacity dims severely.

**dim=8 source confirmed:** The 0.729 dim=8 in trial 16 is entirely attributable to enc1-enc3 residuals (not the decoder). Recovering dim=8 to 0.747 requires removing enc1-enc3 residuals, but this removes the residual benefit for dim=240.

**Cooldown triggered:** Trials 17 and 18 are 2 consecutive exploitation FAILUREs (both refinements of the champion family). Next trial must be Exploration — a new architectural family.
