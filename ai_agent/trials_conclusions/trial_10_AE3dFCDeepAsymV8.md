# Trial 10 — AE3dFCDeepAsymV8 — FAILURE

## Hypothesis

Trials 7–9 established that modifying z_pool position (V5), bottleneck depth (V6), and FC channel count (V7) all degrade V4. A cross-cutting observation from V6 and V7 was that training dynamics differed in suspicious ways: V6 showed uniform dim-agnostic convergence (collapse), and V7 showed rapid early convergence. The common thread: InstanceNorm3d applied at 128×2×8×8 normalizes over only 128 spatial elements per channel. This is below the commonly recommended minimum (~1000 elements) for reliable InstanceNorm statistics, and may cause noisy, high-variance gradient estimates in the bottleneck.

The hypothesis: replacing InstanceNorm3d(128) with GroupNorm(8, 128) in the two bottleneck conv blocks stabilizes normalization statistics. GroupNorm(8, 128) normalizes over 16 channels × 2×8×8 spatial = 2048 elements — well above the instability threshold. More stable gradients should allow the optimizer to find a better local minimum, potentially improving all three dims.

## Implementation

All layers identical to AE3dFCDeepAsymV4 except:

- **bottleneck_conv block 1**: `nn.InstanceNorm3d(128)` → `nn.GroupNorm(8, 128)` (16 channels per group)
- **bottleneck_conv block 2**: `nn.InstanceNorm3d(128)` → `nn.GroupNorm(8, 128)`
- Total params: 1,563,529 at dim=8 (512 extra params from GroupNorm's affine parameters vs InstanceNorm's — negligible)
- All encoder, z_pool, final_down, FC, and decoder layers: identical to V4

## Results

- **R2_dim8:** 0.729131 | **R2_dim60:** 0.753489 | **R2_dim240:** 0.758894
- **avg_validation_R2_mean:** 0.747171
- **delta_vs_champion** (trial avg − champion avg): −0.019477
- **MLflow Run IDs:** fb8a4c64cb224096a138a009ab532773 ca19ef1ba2a240ec8bdfa3192bab61e2 0b7ec57ca71a46fdbe71bbcb3508e077
- **Best epoch dim=8:** 49/300

## Conclusion

The hypothesis did not hold. GroupNorm uniformly degraded all three dims:

- **dim=8:** 0.729 vs V4-champion 0.747 (Δ=−0.018) — moderate regression
- **dim=60:** 0.753 vs V4-champion 0.775 (Δ=−0.022) — moderate regression
- **dim=240:** 0.759 vs V4-champion 0.778 (Δ=−0.019) — moderate regression

The degradation is remarkably uniform across all dims (Δ ≈ −0.02 for all three), in contrast to V6 and V7 which showed dim-specific collapse patterns. This uniformity implies GroupNorm does not reproduce the InstanceNorm instability that was hypothesized — instead it simply provides a different (and worse) normalization for this architecture.

**Why GroupNorm underperforms InstanceNorm here:** InstanceNorm normalizes each channel over its spatial volume — at 2×8×8=128 spatial elements, this forces each channel to have zero mean and unit variance independently. In the context of the bottleneck, this per-channel normalization may act as an implicit regularizer: each channel must carry useful, normalized information, preventing the network from collapsing into a degenerate solution where a few channels carry all the signal. GroupNorm normalizes across channel groups, which does not enforce this per-channel diversity constraint. The loss of per-channel independence may reduce the expressiveness of the 128-channel bottleneck feature, explaining the uniform ~0.02 degradation.

In other words, what appears to be InstanceNorm "instability" (the 128 spatial elements) may actually be a beneficial inductive bias for this architecture. The noisy, high-variance normalization may keep each channel "honest" during training, preventing bottleneck collapse.

**Summary of V4 modification space explored (Trials 7–10):**

| Modification | dim=8 | dim=60 | dim=240 | avg | Verdict |
|---|---|---|---|---|---|
| V4 champion (baseline) | 0.747 | 0.775 | 0.778 | 0.767 | CHAMPION |
| V5: z_pool between enc2/enc3 | 0.721 | 0.707 | 0.776 | 0.735 | FAILURE |
| V6: 3-block bottleneck | 0.731 | 0.727 | 0.728 | 0.729 | FAILURE |
| V7: halved final_down channels | 0.745 | 0.720 | 0.754 | 0.740 | FAILURE |
| V8: GroupNorm in bottleneck | 0.729 | 0.753 | 0.759 | 0.747 | FAILURE |

**Overall campaign conclusion:** AE3dFCDeepAsymV4 (Trial 6) remains the champion with avg R2=0.766648. All 10 trials have been completed. The V4 architecture proved difficult to improve: its z_pool position, bottleneck depth, channel structure, and normalization scheme all appear to be close to locally optimal within the Asym family. Future search directions should consider fundamentally different architectural families or training strategies (e.g. curriculum learning, multi-resolution training, or attention-based global context).
