# Trial 7 — AE3dFCDeepAsymV5 — FAILURE

## Hypothesis

Trial 6 (AE3dFCDeepAsymV4) moved the MaxPool3d(2,1,1) z-halving from after `bottleneck_conv` (V2) to between `enc3` and `enc4`. This improved dim=60 (+0.015 vs V2) while keeping dim=8 stable (−0.001), at the cost of dim=240 (−0.014). The report identified the next direction: moving z_pool even earlier — between `enc2` and `enc3` — to give both `enc3` and `enc4` learned z-compression stages.

The hypothesis: with z_pool2 MaxPool(2,1,1) between `enc2` (16 channels) and `enc3`, both `enc3` and `enc4` each learn to process z pairs through their own MaxPool(2,2,2). This two-stage learned z-compression was expected to push dim=8 closer to AE3dFCDeep's 0.772 while maintaining dim=60 gains from V4.

## Implementation

All layers identical to AE3dFCDeepAsymV4 except the position of the z MaxPool:

- enc1: Conv3DBlock(1→8, ds=False) → 8×32×128×128
- pool1: MaxPool3d(1,2,2) → 8×32×64×64 (unchanged)
- enc2: Conv3DBlock(8→16, ds=True) → 16×16×32×32
- **z_pool2**: MaxPool3d(2,1,1) → 16×8×32×32 [MOVED: one stage earlier than V4]
- enc3: Conv3DBlock(16→32, ds=True) → 32×4×16×16
- enc4: Conv3DBlock(32→64, ds=True) → 64×2×8×8
- bottleneck_conv, final_down, decoder: identical to V4
- Total params: 1,563,017 (unchanged — MaxPool has 0 params)

## Results

- **R2_dim8:** 0.721462 | **R2_dim60:** 0.707073 | **R2_dim240:** 0.775559
- **avg_validation_R2_mean:** 0.734698
- **delta_vs_champion** (trial avg − champion avg): −0.031950
- **MLflow Run IDs:** 265a8be786e240fdbf0664081eebe5e3 dc437f87e3394069aca24c1372b9cc31 14d003e2613d442d9d4cdfe4b4751da6
- **Best epochs:** 30/300 | 34/300 | 47/300

## Training Dynamics

All three dims converged faster than V4 (best epochs 30/34/47 vs 50/40/37). dim=8 converged earliest (best epoch 30), which combined with worse performance indicates the network found a simpler, less expressive solution. The train/val R2 gap is large at dim=60 (train=0.854 vs val=0.707, Δ=0.147) — larger than any previous trial — suggesting overfitting or high variance in the validation set for this architecture. dim=240 showed more stable convergence (best epoch 47, similar to V4's 37).

## Conclusion

The hypothesis did not hold. Moving z_pool one stage earlier (between enc2 and enc3 instead of enc3/enc4) degraded all three dims:

- **dim=8:** 0.721 vs V4-champion 0.747 (Δ=−0.026) — significant regression
- **dim=60:** 0.707 vs V4-champion 0.775 (Δ=−0.068) — catastrophic regression
- **dim=240:** 0.776 vs V4-champion 0.778 (Δ=−0.002) — essentially unchanged

The dim=60 result (0.707) is the second-worst single-dim result across all trials (after T5's 0.688). The pattern across the z_pool position series is now clear:

| z_pool position | dim=8 | dim=60 | dim=240 | avg |
|-----------------|-------|--------|---------|-----|
| After bottleneck (V2) | 0.748 | 0.760 | 0.792 | 0.766 |
| After enc3 (V4) | 0.747 | 0.775 | 0.778 | 0.767 |
| After enc2 (V5) | 0.721 | 0.707 | 0.776 | 0.735 |

Moving z_pool from after `bottleneck_conv` to after `enc3` was a net improvement (better dim=60 at the cost of dim=240). Moving it one stage further to after `enc2` is a dramatic failure at dim=60 with no benefit at dim=8.

The mechanistic explanation for dim=60's collapse: at the 16-channel stage (after enc2), individual channels encode low-level, spatially local features. Max-selecting one z-slice over the other at this stage discards more semantically rich information than doing so at the 32-channel stage (V4, after enc3) or 128-channel stage (V2, after bottleneck). At dim=60 — which requires intermediate compression — the encoder needs richer z-features to populate 60 codes; V5 eliminates half the z-information before the encoder has had enough stages to abstract it into redundant representations, causing information collapse.

The dim=8 regression (vs V4) also contradicts the hypothesis. Despite having two learned z-compression stages (enc3 and enc4), V5 performs worse than V4 at dim=8. This suggests that the MaxPool position in V4 — after enc3 where channels are already at 32 and features more abstract — is optimal for the small-latent-dim case, not just for large dims.

**Key takeaway:** The z_pool position in V4 (between enc3 and enc4) is the empirical optimum within the Asym family. Moving it in either direction (earlier to enc2/enc3, later to after bottleneck) either reduces the benefit or causes regression. Future trials should explore different architectural families or modifications that don't change z_pool position.
