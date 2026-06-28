# Trial 14 — AE3dAsymResidualV3 — FAILURE

## Hypothesis
Trial 13 showed that removing z_pool3 recovers dim=240 but the resulting 4096 FC collapsed dim=60. The proposed fix: keep flattened_size=2048 by inserting a learned z-compression conv (Conv3d(128,128,k=(2,1,1),s=(2,1,1))+IN+ReLU) after bottleneck_conv instead of the MaxPool z_pool3 between enc3/enc4. This would let enc4 see richer z=8 features while controlling FC size.

## Implementation
- No z_pool3 between enc3/enc4 (enc4 gets z=8 inputs).
- bottleneck_conv: 64×4×8×8 → 128×4×8×8.
- z_compress: Conv3d(128,128,k=(2,1,1),s=(2,1,1))+IN+ReLU: 128×4×8×8 → 128×2×8×8 (learned z halving at bottleneck).
- final_down Conv3d(k=2,s=2): 128×2×8×8 → 128×1×4×4, flattened_size=2048.
- Decoder: initial_up → 128×2×8×8, z_expand Upsample(2,1,1) → 128×4×8×8, then dec1-dec4 as in trial 12.

## Results
- **R2_dim8:** 0.696706 | **R2_dim60:** 0.701617 | **R2_dim240:** 0.776631
- **avg_validation_R2_mean:** 0.724985
- **delta_vs_champion** (trial avg − champion avg): −0.041663
- **MLflow Run IDs:** f80bd567947447309d985e3b84ecaabe bf5fce8ad2de4e389c1e541b24da2fa5 b496ba90d80e4e58802b910072ae75e5
- **Best epochs:** 25/55 | 16/46 | ~24/54

## Training Dynamics
All dims converged with early stopping around ep 46–55. dim=240 was stable (val loss 0.000663). dims 8 and 60 both showed high val losses (0.000830, 0.000808 respectively) — notably worse than trial 12.

## Conclusion
The learned z-compress failed badly at dim=8 and dim=60. The likely mechanism: the z_compress conv introduces a non-trivial transformation immediately before final_down → FC. At small latent dims (8, 60), the FC already has limited capacity — adding an extra learned transformation at the bottleneck increases the optimization difficulty (more layers to optimise jointly with limited data and capacity). By contrast, MaxPool is deterministic and imposes no additional learnable parameters, making the downstream FC easier to learn. The asymmetric MaxPool z_pool3 in trial 12 (placed between enc3/enc4, not at the bottleneck) was the correct design because it compresses z before the final encoding stages, giving the encoder layers more epochs to adapt. Moving z-compression to the bottleneck (after bottleneck_conv) interrupts the direct path from the last conv features to the FC. Conclusion: z_pool3's position in V4 (mid-encoder) is architecturally important, not incidental.
