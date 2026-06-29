# Trial 22 — AE3dAsymResSeparableV3 — FAILURE

## Hypothesis
Trial 21 champion (V2) keeps enc4 as a plain SeparableConv3DBlock — the only encoder stage without a residual shortcut. The hypothesis: completing the fully-residual encoder chain (enc1-4 all ResSeparableConv3DBlock) would provide an unbroken gradient path from all encoder stages to the bottleneck, expected to push dim=8 above 0.792. The enc4 residual shortcut adds Conv3d(32,64,1) (~2K params), negligible cost.

## Implementation
- Identical to V2 except:
  - enc4: **ResSeparableConv3DBlock(32, 64, downsample=True)** (was SeparableConv3DBlock)
- All other components unchanged: enc1-enc3 ResSep, dec1-3 ResUpSep, dec4_conv ResSep, V4 anisotropic pooling, bottleneck/FC identical

## Results
- **R2_dim8:** 0.761688 | **R2_dim60:** 0.804424 | **R2_dim240:** 0.778864
- **avg_validation_R2_mean:** 0.781659
- **delta_vs_champion** (trial avg − champion avg): −0.029880
- **MLflow Run IDs:** f1be78ae1aad49f4829ef7d508fed1a8 c9807552d39246b3935c505b928010bb 3d28705264164f65b812b7b3e4ab0c39

## Training Dynamics
All dims converged but at worse minima than V2. Dim=240 was particularly affected (0.817→0.779, −0.038), suggesting the residual shortcut in enc4 actively disrupts the high-capacity latent representation. Dim=8 also degraded (0.792→0.762, −0.030), opposite to the hypothesis.

## Conclusion
**Hypothesis refuted.** The plain enc4 in V2 is not a weakness — it is a deliberate architectural feature. The residual shortcut in enc4 systematically degraded all three dims.

| Dim | V2 Champion | V3 | Delta |
|-----|------------|-----|-------|
| dim=8 | 0.792 | 0.762 | −0.030 |
| dim=60 | 0.826 | 0.804 | −0.022 |
| dim=240 | 0.817 | 0.779 | −0.038 |
| **avg** | **0.812** | **0.782** | **−0.030** |

**Mechanism — why enc4 residual hurts:**

The enc4 stage operates after z_pool3 (MaxPool3d(2,1,1)), which has already halved the z-depth. At this point the spatial feature map is 32×4×16×16. The residual shortcut in enc4 carries a direct 32→64 channel identity path through a 1×1×1 conv that bypasses the DW+PW feature factorization. This shortcut:

1. **Interferes with forced feature reduction**: the plain SeparableConv3DBlock(32→64) is a bottleneck that forces full channel re-factorization from 32 to 64. The residual shortcut creates an "easy path" that allows the network to skip this factorization, reducing the quality of the 64-channel representation entering the bottleneck.

2. **Disrupts the V4 asymmetry**: the V4 design deliberately uses plain enc4 (no residual) as a "free transformation" stage after z_pool3. The asymmetry between enc1-3 (residual/identity shortcuts) and enc4 (free transformation) creates a two-stage encoder: early stages (enc1-3) focus on robust feature extraction with stable gradients, while enc4 focuses on aggressive compression into the bottleneck spatial dimensions. The residual shortcut in enc4 collapses this specialization.

3. **Particularly harmful for dim=240**: the high-capacity latent space needs the enc4 stage to independently discover the optimal 64-channel bottleneck representation. The residual shortcut constrains enc4 output to be a weighted combination of shortcut (32→64 projection) and separable path — this constraint is harmful when the bottleneck needs maximum freedom.

**Key finding:** The enc4 boundary (between the residual enc1-3 chain and the free enc4 stage) is architecturally load-bearing. The plain SeparableConv3DBlock at enc4 must be preserved in all future V-series variants.
