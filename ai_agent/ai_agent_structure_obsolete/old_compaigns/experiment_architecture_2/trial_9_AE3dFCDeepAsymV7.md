# Trial 9 — AE3dFCDeepAsymV7 — FAILURE

## Hypothesis

Trials 6 (V4), 7 (V5), and 8 (V6) established that: the z_pool position in V4 is optimal, moving it earlier hurts (V5), and adding bottleneck depth causes InstanceNorm instability (V6). The next unexplored dimension was the FC compression ratio. V4 uses final_down Conv3d(128→128, k=2, s=2), producing a 128×1×4×4 = 2048-element flattened vector. At dim=8, this means 256× compression through the FC; at dim=60, 34×; at dim=240, 8.5×.

The hypothesis: halving the channels in final_down (128→64, flattened_size 2048→1024) reduces the compression ratio for all dims (128× → 64× for dim=8; 34× → 17× for dim=60; 8.5× → 4.3× for dim=240). A lower compression ratio should benefit all dims by making the FC bottleneck less information-lossy, improving reconstruction fidelity across the board.

## Implementation

All layers identical to AE3dFCDeepAsymV4 except:

- **final_down**: `Conv3d(128, 64, kernel_size=2, stride=2)` → shape 64×1×4×4, flattened_size=1024 (vs 128×1×4×4=2048 in V4)
- **fc_enc**: `nn.Linear(1024, latent_dim)` (vs 2048)
- **fc_dec**: `nn.Linear(latent_dim, 1024)` (vs 2048)
- **initial_up**: `ConvTranspose3d(64, 128, kernel_size=2, stride=2)` → restores 128×2×8×8
- All encoder (enc1–enc4, z_pool3), bottleneck_conv (2 blocks, 64→128→128), and decoder (z_up, dec1–dec4, final_conv) layers: identical to V4
- Total params: ~1.25M at dim=8 (vs V4's 1.56M)

## Results

- **R2_dim8:** 0.744966 | **R2_dim60:** 0.720271 | **R2_dim240:** 0.753542
- **avg_validation_R2_mean:** 0.739593
- **delta_vs_champion** (trial avg − champion avg): −0.027055
- **MLflow Run IDs:** 3a37606b17fd4862addc90b79ac33ef4 804d67b857d041b7a16e2505c4ed8542 7b49e7059e6b4b5eb9d502f403141326
- **Best epochs:** dim=8 (not explicitly logged) | dim=60: 54/300 | dim=240: (not explicitly logged)

## Training Dynamics

dim=60 converged fastest among the three dims: best at epoch 54 with early stopping at epoch 84. The learning rate decayed progressively (5e-5 → 1.56e-6) without finding a better optimum after epoch 54. This rapid convergence to a plateau is similar to V6's pattern, suggesting the bottleneck modification also caused premature convergence.

## Conclusion

The hypothesis did not hold. Halving the final_down channels degraded all three dims relative to V4:

- **dim=8:** 0.745 vs V4-champion 0.747 (Δ=−0.002) — essentially unchanged
- **dim=60:** 0.720 vs V4-champion 0.775 (Δ=−0.055) — significant regression
- **dim=240:** 0.754 vs V4-champion 0.778 (Δ=−0.024) — notable regression

dim=8 was nearly unaffected, consistent with the hypothesis (at 64× compression even the halved feature is sufficient for 8 codes). However, dim=60 regressed dramatically — worse than V6 (0.727) and approaching V5 (0.707) territory.

**Why halving channels hurts dim=60:** The 64-channel feature map at 1×4×4 = 1024 spatial-channel elements must linearly project to 60 latent codes. While 1024→60 is a large compression, the 64 channels may not span a rich enough representation to independently encode 60 semantically distinct modes of variation. By contrast, V4's 128-channel feature (2048 elements) provides twice the channel diversity, from which 60 independent codes can be more readily extracted. At dim=8 this distinction doesn't matter (8 codes are easily extracted from 64 channels), but at dim=60 the reduced channel expressiveness becomes a binding constraint.

**Paradox:** The change reduced the compression ratio (making FC bottleneck less information-lossy), yet performance degraded. This reveals that compression ratio is not the limiting factor in V4 — the limiting factor is the richness of the spatial channel representation before the FC. V4's 128-channel pre-FC feature is richer than V7's 64-channel feature, even though V4 compresses it more aggressively.

**Summary of V4 modification space explored (Trials 6–9):**

| Modification | dim=8 | dim=60 | dim=240 | avg | Verdict |
|---|---|---|---|---|---|
| V4 champion (baseline) | 0.747 | 0.775 | 0.778 | 0.767 | CHAMPION |
| V5: z_pool earlier | 0.721 | 0.707 | 0.776 | 0.735 | FAILURE |
| V6: 3-block bottleneck | 0.731 | 0.727 | 0.728 | 0.729 | FAILURE |
| V7: halved final_down | 0.745 | 0.720 | 0.754 | 0.740 | FAILURE |

**Key takeaway:** The V4 architecture is robust against modifications in its explored space. Moving z_pool, adding bottleneck depth, and halving the pre-FC channels all degrade performance. The bottleneck normalization (InstanceNorm at 128×2×8×8) remains a hypothetical weak point — switching to GroupNorm (which normalizes over channel groups rather than spatial dimensions) may enable more stable optimization without altering the spatial structure that makes V4 successful.
