# Trial 13 — AE3dAsymResidualV2 — FAILURE

## Hypothesis
Trial 12 (AE3dAsymResidual) showed that z_pool3=(2,1,1) hurts dim=240 (0.747 vs trial 11's 0.792 with isotropic pooling). The fix: keep pool1=(1,2,2) for dim=8 protection, remove z_pool3 to recover dim=240. Without z_pool3, enc4 outputs 64×4×8×8 (z=4 instead of 2), and final_down produces 128×2×4×4, doubling flattened_size to 4096.

## Implementation
- Identical to AE3dAsymResidual except z_pool3 removed.
- enc4 now processes 32×8×16×16 → 64×4×8×8 (z stays at 4 after enc4's isotropic MaxPool(2,2,2)).
- bottleneck_conv: 64×4×8×8 → 128×4×8×8.
- final_down Conv3d(k=2,s=2): 128×4×8×8 → 128×2×4×4.
- flattened_size = 4096 (vs 2048 in all prior AsymResidual variants).
- Decoder mirrors: initial_up ConvTranspose(k=2,s=2) restores 128×4×8×8; no z_up needed.

## Results
- **R2_dim8:** 0.745900 | **R2_dim60:** 0.739301 | **R2_dim240:** 0.791823
- **avg_validation_R2_mean:** 0.759008
- **delta_vs_champion** (trial avg − champion avg): −0.007640
- **MLflow Run IDs:** de86ad2203f141408bb05b084a994438 c364c74cb2f44f518a6f9d6423a04c99 3770703ccd544744a0dd498a3a64b02e
- **Best epochs:** 20/50 | 26/56 | ~28/58

## Training Dynamics
All dims showed stable convergence with early stopping around ep 50–58. dim=240 val loss (0.000622) was the best of the three dims, consistent with the hypothesis that removing z_pool3 benefits large-capacity dims.

## Conclusion
The hypothesis was half-validated: removing z_pool3 did recover dim=240 (0.792, matching trial 11's isotropic result). However, the doubled FC size (4096) caused a severe collapse at dim=60 (0.739 vs 0.799 in trial 12 with 2048). The larger FC introduces more parameters that are hard to optimise with only 100 training patients — the model likely overfits in the FC layers. dim=8 was largely unaffected (0.746 vs 0.740 in trial 12). The key tension: removing z_pool3 to help dim=240 forces a larger FC that hurts dim=60. To resolve this, a bottleneck compression before the FC (e.g., an extra Conv3d(128,128,k=(2,1,1)) to halve z from 4→2) could restore flattened_size=2048 while still benefiting from the z_pool3 removal earlier in the encoder.
