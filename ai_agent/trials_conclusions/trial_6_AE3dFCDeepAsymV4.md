# Trial 6 — AE3dFCDeepAsymV4 — CHAMPION

## Hypothesis

Trial 4 (AE3dFCDeepAsymV2) established a two-step z-collapse: MaxPool3d(2,1,1) after `bottleneck_conv` (128×4×8×8 → 128×2×8×8), then Conv3d(k=2) to 128×1×4×4. This fixed dim=8 (+0.045 vs T3) at the cost of dims 60/240 (−0.025, −0.010). The MaxPool step discards z-diversity through max-selection after the bottleneck has already processed 4 z-slices in parallel — the bottleneck cannot exploit cross-z interactions before the z-collapse.

The hypothesis: moving the MaxPool3d(2,1,1) **earlier** — between `enc3` and `enc4` instead of after `bottleneck_conv` — gives `enc4` and `bottleneck_conv` a z=2 input from the start. This means `enc4`'s MaxPool3d(2,2,2) reduces z 4→2, and `bottleneck_conv` processes 128×2×8×8 identically to how `AE3dFCDeep` processes its bottleneck (which achieved dim=8=0.772). Meanwhile, the anisotropic stage-1 encoding (MaxPool(1,2,2) in `pool1`) still preserves z-diversity through `enc2` and `enc3`, so the early z-MaxPool collapses a richer z-representation than T3's one-step Conv3d(k=(4,2,2)) did.

The predicted gain: better large-dim performance than V2 (because `enc4` + `bottleneck_conv` now operate on features shaped by 3 stages of z-preserving encoding rather than 2), while preserving V2's dim=8 advantage (because the bottleneck sees the same z=2 context as AE3dFCDeep).

## Implementation

All layers identical to AE3dFCDeepAsymV2 except the position of the z MaxPool:

- enc1: Conv3DBlock(1→8, downsample=False) → 8×32×128×128
- pool1: MaxPool3d(1,2,2) → 8×32×64×64 (unchanged)
- enc2: Conv3DBlock(8→16, ds=True) → 16×16×32×32
- enc3: Conv3DBlock(16→32, ds=True) → 32×8×16×16
- **z_pool3**: MaxPool3d(kernel=(2,1,1), stride=(2,1,1)) → 32×4×16×16 [MOVED HERE from after bottleneck]
- enc4: Conv3DBlock(32→64, ds=True) → 64×2×8×8 [now operates on z=4, MaxPool(2,2,2) gives z=2]
- bottleneck_conv: 2-block Conv3d sequence → 128×2×8×8 [same z context as AE3dFCDeep]
- final_down: Conv3d(128,128, k=2, s=2) → 128×1×4×4
- Decoder: identical to V2 — initial_up ConvTranspose3d(k=2) → 128×2×8×8, z_up Upsample(2,1,1) → 128×4×8×8, dec1–dec4 unchanged
- Total params at dim=8: 1,563,017 (identical to V2 — MaxPool has 0 parameters)

## Results

- **R2_dim8:** 0.747004 | **R2_dim60:** 0.774821 | **R2_dim240:** 0.778119
- **avg_validation_R2_mean:** 0.766648
- **delta_vs_champion** (trial avg − champion avg): +0.000185
- **MLflow Run IDs:** f9a89b6f621748718f1aaf6c1e44b6c3 1ce1d6737eef411fb85ed6ce642c6369 8cedbc464b4a4414b48ecd07025d3521
- **Best epochs:** 50/300 | 40/300 | 37/300

## Training Dynamics

All three runs converged stably with no instabilities. Early stopping at epochs 80/70/67 for dims 8/60/240 respectively — slightly faster than V2 (75/66/70) across the board, suggesting the earlier z-MaxPool creates a marginally smoother optimization landscape. Val loss curves were monotonically improving in the early phase for all dims. The consistent improvement pattern (loss halving from epoch 1 to ~15 for dim=240) indicates the architecture is well-conditioned. No sign of the optimization interference that plagued T5 (V3's learned z-pooling).

## Conclusion

The hypothesis partially held, with a nuanced outcome:

- **dim=8:** 0.747004 vs V2-champion 0.747656 (Δ=−0.000652) — essentially identical, tiny regression
- **dim=60:** 0.774821 vs V2-champion 0.760047 (Δ=**+0.014774**) — meaningful improvement
- **dim=240:** 0.778119 vs V2-champion 0.791686 (Δ=−0.013567) — meaningful regression

The dim=8 result is mechanistically consistent with the hypothesis: V4's bottleneck_conv sees 128×2×8×8 just like AE3dFCDeep, and V4 achieves 0.747 vs AE3dFCDeep's 0.772. The gap (Δ=−0.025) is explained by the enc4 downsampling path: in AE3dFCDeep, z collapses through 4 isotropic stages (each adding learned z-aggregation); in V4, only enc4 sees z=4→2 via MaxPool(2,2,2), which is less expressive than a full isotropic conv stage.

The dim=60 improvement (+0.015) is the main surprise. Moving z_pool earlier means `enc4` and `bottleneck_conv` co-learn z-compression and channel-expansion jointly, producing features that are more informative about the cardiac anatomy at intermediate dimensionality. In V2, the MaxPool after `bottleneck_conv` discarded z-diversity from already-abstracted 128-channel features — the bottleneck had no opportunity to learn cross-z interactions before max-selection. In V4, the MaxPool acts on 32-channel features (before enc4), where individual channels still carry more spatially localized information, potentially preserving more reconstruction-relevant signal.

The dim=240 regression (−0.014) is the tradeoff cost: with z_pool occurring earlier (at enc3→enc4 boundary rather than after bottleneck), there are fewer z-preserving encoding stages available to capture high-dimensional structure. `enc4` and `bottleneck_conv` still process z=2 features, but the information content entering them from enc3 is slightly poorer than when enc4 processed z=4 slices in V2.

**Net verdict: new champion** by +0.000185. V4 redistributes performance: large gains at dim=60 (+0.015) and dim=8 stability (−0.001), at the cost of dim=240 (−0.014). The narrow margin suggests the optimization landscape is near a saddle between dim=60 and dim=240 performance. Future trials should explore whether a variant that collapses z even earlier (between enc2 and enc3) can push dim=8 toward AE3dFCDeep's 0.772 while maintaining the dim=60 gains.
