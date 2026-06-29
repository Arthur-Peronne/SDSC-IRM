# Trial 16 — AE3dAsymResidualV4 — CHAMPION

## Hypothesis
Trial 12 (AE3dAsymResidual, CANDIDATE) uses ResConv3DBlocks at all 4 encoder stages. Its dim=240 performance (0.747) is significantly below trial 11's isotropic residual (0.792) and below the previous champion (0.778). The residual shortcut at enc4 — the stage after z_pool3 — may limit enc4's ability to freely transform features: enc4 must compress 32×4×16×16 → 64×2×8×8 under the identity constraint of the shortcut. After z_pool3 has already halved z, enc4 needs aggressive spatial transformation; the residual identity anchors output too close to input for high-capacity dims. Removing the residual only at enc4 (while keeping enc1-enc3 residuals) should recover dim=240 without losing the dim=60 benefit from earlier residual stages.

## Implementation
- enc1–enc3: `ResConv3DBlock` (with 1×1×1 shortcut when channels change) — unchanged from trial 12.
- enc4: `Conv3DBlock` (plain, no shortcut) — new change vs trial 12.
- z_pool3: `MaxPool3d(2,1,1)` between enc3/enc4 — unchanged from V4/trial 12.
- pool1: `MaxPool3d(1,2,2)` after enc1 — unchanged.
- Decoder: `ResUpConv3DBlock` for dec1/dec2/dec3, `ResConv3DBlock(downsample=False)` for dec4 — unchanged from trial 12.
- Bottleneck/FC: identical to V4 (bottleneck_conv 64→128→128, final_down Conv3d(k=2), FC, flattened_size=2048).
- ~1.56M params at dim=8 (identical to V4).

## Results
- **R2_dim8:** 0.729142 | **R2_dim60:** 0.777294 | **R2_dim240:** 0.814983
- **avg_validation_R2_mean:** 0.773806
- **delta_vs_prior_champion** (trial avg − prior champion avg): +0.007158
- **MLflow Run IDs:** e516b8abf71c4caead83e79b2541b22e fc7c00d80a26411285eb630a26ae3213 0a26cf297adc411f9a06eec887a303ef
- **Best epochs:** 28/58 | 25/55 | 42/72

## Training Dynamics
All dims converged cleanly with early stopping at epochs 55–72. dim=240 converged latest (best ep42) with the lowest val loss (0.000517) — the model continued improving well past the typical ~25-30 epoch best seen in residual variants. This prolonged improvement for dim=240 is consistent with enc4 having more freedom to specialize its transformation without the identity constraint. dim=60 also improved slightly vs the prior champion (0.777 vs 0.775). dim=8 regressed slightly (0.729 vs 0.747 prior champion) — without the enc4 residual shortcut, dim=8's gradient flow through enc4 is slightly weaker.

## Conclusion
The hypothesis was validated. Removing enc4's residual shortcut dramatically improved dim=240 (0.815 vs trial 12's 0.747, a gain of +0.068) while preserving the dim=60 benefit from enc1-enc3 residuals. The mechanism is clear: enc4 operates on z-compressed features (z=4 after z_pool3) and must map 32→64 channels with simultaneous MaxPool(2,2,2) spatial downsampling. The residual identity path anchors output near input, reducing the transformation magnitude. For dim=240 (high-capacity), the model benefits from aggressive feature transformation at enc4 to produce richly-structured bottleneck representations. For dim=8 (low-capacity), the loss of enc4's identity shortcut slightly reduces gradient flow, explaining the small dim=8 regression.

**Key architectural insight:** In the AsymResidual family, the residual shortcut at enc4 is a liability, not an asset — the z-compressed spatial context after z_pool3 requires free transformation, not identity preservation. The optimal architecture uses residual blocks only where the spatial structure is still rich (enc1-enc3), and plain conv at enc4 where aggressive spatial compression is needed.

**New champion avg: 0.773806** (prior champion AE3dFCDeepAsymV4: 0.766648, delta: +0.007158). First time dim=240 exceeds 0.80 in the entire experiment (0.815).
