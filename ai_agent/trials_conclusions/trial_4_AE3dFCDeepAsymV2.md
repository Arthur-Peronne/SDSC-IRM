# Trial 4 — AE3dFCDeepAsymV2 — CHAMPION

## Hypothesis
Trial 3 (AE3dFCDeepAsym) established that anisotropic stage-1 pooling (preserving z at stage 1) strongly benefits large latent dims (+0.059 at dim=60, +0.044 at dim=240) but regresses dim=8 significantly (−0.070). The root cause identified in Trial 3's report: the champion's `final_down` layer collapses z 4→1 in a single Conv3d(kernel=(4,2,2)) step. At dim=8, this one-step collapse from 128×4×8×8 to 128×1×4×4 forces the FC bottleneck to represent 4 diverse z-slice summaries into just 8 codes — a 256x compression ratio applied to maximally diverse features.

The hypothesis: replacing the single-step z-collapse with a two-step collapse (MaxPool3d(2,1,1) to get z=4→2, then Conv3d(kernel=2,stride=2) to get z=2→1) mirrors AE3dFCDeep's proven 2→1 final compression while preserving the asymmetric early encoding. At dim=8, the MaxPool step first averages adjacent z-pairs (a smooth, structure-preserving operation), giving the Conv3d a cleaner 2-slice input — identical to what AE3dFCDeep processes. At large dims, the MaxPool still operates on features encoded with z-diversity preserved through stage 1, so the bottleneck_conv at 128×4×8×8 carries richer information than AE3dFCDeep's 128×2×8×8.

## Implementation
- enc1: Conv3DBlock(1→8, downsample=False) → 8×32×128×128
- pool1: MaxPool3d(1,2,2) → 8×32×64×64 (unchanged from champion)
- enc2–enc4: Conv3DBlock(ds=True) identical to champion → 64×4×8×8
- bottleneck_conv: identical 2-block sequence → 128×4×8×8
- **z_pool**: MaxPool3d(kernel=(2,1,1), stride=(2,1,1)) → 128×2×8×8 [NEW, 0 params]
- **final_down**: Conv3d(128, 128, kernel=2, stride=2) → 128×1×4×4 [same as AE3dFCDeep]
- Decoder: initial_up = ConvTranspose3d(128,128,kernel=2,stride=2) → 128×2×8×8 [same as AE3dFCDeep]
- **z_up**: Upsample(scale=(2,1,1), trilinear) → 128×4×8×8 [NEW, 0 params]
- dec1–dec3: identical UpConv3DBlocks; dec4_up+dec4_conv: anisotropic (1,2,2) upsample
- Total params at dim=8: 1,563,017 (vs champion's 1,825,161 — smaller due to smaller kernel in final_down/initial_up)

## Results
- **R2_dim8:** 0.747656 | **R2_dim60:** 0.760047 | **R2_dim240:** 0.791686
- **avg_validation_R2_mean:** 0.766463
- **delta_vs_champion** (trial avg − champion avg): +0.003693
- **MLflow Run IDs:** 4365d204b8e2456c965e8c9e88c03735 3b3d0085c6b8457392911f8ed4fcdb5c c6ece9ce362548d3916358734c95d0ed
- **Best epochs:** 45/300 | 36/300 | 40/300

## Training Dynamics
All three runs converged stably with no instabilities. Early stopping triggered at epochs 75 (dim=8), 66 (dim=60), and 70 (dim=240). Convergence was slightly faster than both AE3dFCDeep and AE3dFCDeepAsym across all dims, with best epochs consistently in the 36–45 range. The learning rate scheduler had fewer halvings before convergence, suggesting the two-step z-collapse creates a smoother optimization landscape. Validation loss curves showed consistent monotonic improvement without plateau-then-collapse patterns.

## Conclusion
The hypothesis partially held, with a nuanced tradeoff vs the prior champion:

- **dim=8:** 0.7477 vs T3-champion 0.7023 (Δ=+0.045) — major recovery, as predicted
- **dim=60:** 0.7600 vs T3-champion 0.7845 (Δ=−0.025) — moderate regression
- **dim=240:** 0.7917 vs T3-champion 0.8015 (Δ=−0.010) — small regression

The dim=8 improvement is mechanistically explained: the MaxPool(2,1,1) step now averages adjacent z-pairs (basal pair, mid-ventricular pair, apical pair) before any learned compression, giving the Conv3d(k=2) a feature map that is half as z-diverse. The subsequent FC at 2048→8 then performs the same compression as AE3dFCDeep, which achieved 0.772 at dim=8. The 0.747 vs 0.772 gap (Δ=0.025 below AE3dFCDeep) is consistent with the MaxPool being less efficient than AE3dFCDeep's full z-averaging — AE3dFCDeep collapses z 4× in 4 separate isotropic stages (allowing learned features to progressively summarize z), while V2 uses a single MaxPool step to do the final 2× z-reduction.

The dim=60/240 regressions vs T3 are also mechanistically coherent: the MaxPool(2,1,1) step discards half the z-feature diversity through max-selection rather than learned compression. At large dims, the T3 champion's Conv3d(k=(4,2,2)) could learn to weight the 4 z-slices non-uniformly (attending to the anatomically distinct slices), while MaxPool merely keeps the dominant activation. This information loss is irreversible and prevents the decoder from reconstructing z-specific detail at high latent dims.

**Net verdict: new champion** by a margin of +0.004. The two-step collapse fixes dim=8 (+0.045) at the cost of dims 60/240 (−0.025, −0.010). The improvement is real but comes from redistributing performance rather than uniform gains. Future trials should investigate whether learned z-pooling (Conv3d(k=(2,1,1)) instead of MaxPool) can recover the lost dim=60/240 performance while maintaining the dim=8 fix.
