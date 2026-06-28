# Trial 8 — AE3dFCDeepAsymV6 — FAILURE

## Hypothesis

Trials 5, 6, and 7 showed that z_pool position (V3, V4, V5) and z_pool mechanism (V3) are the primary levers in the Asym family. V4 is the optimal z_pool configuration. The next unexplored dimension was bottleneck depth: V4 has a 2-block bottleneck (64→128→128 channels). Adding a 3rd block (128→128) would give the encoder more non-linear processing capacity at the 128×2×8×8 feature level before final_down.

The hypothesis: more depth in the bottleneck allows the network to learn more expressive, structured compressed features, improving reconstruction fidelity across all latent dims. The 2048-element flattened vector would encode richer spatial patterns.

## Implementation

All layers identical to AE3dFCDeepAsymV4 except:

- **bottleneck_conv**: 3 blocks instead of 2:
  - Conv3d(64→128, k=3, p=1) + InstanceNorm3d(128) + ReLU
  - Conv3d(128→128, k=3, p=1) + InstanceNorm3d(128) + ReLU
  - Conv3d(128→128, k=3, p=1) + InstanceNorm3d(128) + ReLU ← new
- Everything else (enc1–enc4, z_pool3, final_down, FC, decoder): identical to V4
- Total params: ~2.0M at dim=8 (+28% vs V4's 1.56M, +442K from the extra conv block)

## Results

- **R2_dim8:** 0.731369 | **R2_dim60:** 0.727277 | **R2_dim240:** 0.727881
- **avg_validation_R2_mean:** 0.728842
- **delta_vs_champion** (trial avg − champion avg): −0.037806
- **MLflow Run IDs:** 8b1826782c8d4788b3d68966db28b7b3 cfc0e7786548462b8fc3bcfae8f4e038 54bc013d6f4947d4a81059b69267beed
- **Best epochs:** 41/300 | 52/300 | 50/300

## Training Dynamics

Convergence speed was similar to V4 (best epochs 41/52/50 vs 50/40/37 for V4). Notably, all three dims converged to nearly identical val losses (~0.000746–0.000767) — a collapse toward a similar loss value regardless of latent dim. This is atypical: in V4 and V2, dim=240 converged to lower val loss than dim=8. The convergence uniformity suggests the 3-block bottleneck drove the network to a degenerate local minimum that is not latent-dim-sensitive.

## Conclusion

The hypothesis did not hold. The deeper bottleneck uniformly degraded all three dims:

- **dim=8:** 0.731 vs V4-champion 0.747 (Δ=−0.016)
- **dim=60:** 0.727 vs V4-champion 0.775 (Δ=−0.048)
- **dim=240:** 0.728 vs V4-champion 0.778 (Δ=−0.050)

The near-identical validation R2 across all three dims (0.731/0.727/0.728) is the most striking result — it suggests the model is effectively ignoring latent capacity (the network learns approximately the same reconstruction regardless of whether it has 8 or 240 latent codes). This is a strong signal of information collapse in the bottleneck.

Two mechanisms explain the failure:

1. **InstanceNorm instability compounding:** At the 128×2×8×8 feature level, InstanceNorm3d normalizes over 2×8×8=128 spatial elements per channel. This is already borderline unstable for a 2-block bottleneck (each iteration of V4's training must cope with noisy batch statistics). Adding a 3rd InstanceNorm at this spatial scale triples the normalization noise in the bottleneck path, creating an unstable gradient landscape. The network cannot reliably learn from the 3rd block's gradients.

2. **Optimization depth / gradient flow:** With 3 bottleneck blocks + final_down + FC, the gradient must traverse more non-linearities. In a small spatial volume (2×8×8), the expressive capacity of 3 conv blocks is limited (the 3rd block sees the same 2×8×8 spatial pattern with no way to increase receptive field), while the optimization cost increases. The extra depth likely causes the optimizer to settle at a shallower local minimum.

**Key takeaway:** At the 128×2×8×8 spatial scale, 2 conv blocks is the effective depth limit. Adding a 3rd block induces InstanceNorm instability and optimization degradation without capacity gains. Future trials should change the *width* or *connectivity* of the bottleneck rather than its depth, or reduce the flattened representation size to make the FC bottleneck compression ratio more tractable.
