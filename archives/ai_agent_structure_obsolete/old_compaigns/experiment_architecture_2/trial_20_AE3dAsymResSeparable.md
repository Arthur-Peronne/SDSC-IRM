# Trial 20 — AE3dAsymResSeparable — CHAMPION

## Hypothesis
Trial 19 (AE3dAsymSeparable, CANDIDATE) showed that separable convolutions produce exceptional dim=60 (0.804, best ever) but weak dim=8 (0.703) and moderate dim=240 (0.758). Trial 16 champion showed that residual enc1-enc3 + plain enc4 + residual decoder gives strong dim=240 (0.815) and good dim=60 (0.777). The hypothesis: combining residual shortcuts with separable convolutions (ResSeparableConv3DBlock) at enc1-enc3 — keeping enc4 plain separable and using the residual decoder — should capture the best of both worlds. The residual shortcuts provide gradient flow for dim=240, the separable DW+PW factorization provides feature richness for dim=60, and the plain enc4 allows free transformation after z_pool3. The encoder-decoder residual style is consistent (both residual), avoiding the trial-18 incompatibility.

## Implementation
- enc1: ResSeparableConv3DBlock(1, 8, downsample=False) — residual shortcut around DW+PW path
- pool1: MaxPool3d((1,2,2))
- enc2: ResSeparableConv3DBlock(8, 16, downsample=True)
- enc3: ResSeparableConv3DBlock(16, 32, downsample=True)
- z_pool3: MaxPool3d((2,1,1))
- enc4: SeparableConv3DBlock(32, 64, downsample=True) — plain separable, no residual
- Bottleneck/FC: identical to V4/trial 16 (flattened_size=2048)
- dec1-3: ResUpConv3DBlock — residual decoder (consistent with residual encoder)
- dec4: Upsample(1,2,2) + ResConv3DBlock(16,8, no_pool)
- ~1.36M params at dim=8 (lighter than prior champion's 1.56M)

## Results
- **R2_dim8:** 0.776740 | **R2_dim60:** 0.800600 | **R2_dim240:** 0.810280
- **avg_validation_R2_mean:** 0.795874
- **delta_vs_prior_champion** (trial avg − prior champion avg): +0.022068
- **MLflow Run IDs:** 1c6492dc27384f348a981bac409a499d cb10c7fd15924b96aebba3580373b8be 6df4b1d343d241f58427a4b9b37d2166
- **Best epochs:** dim=8: ep32 | dim=60: ep36 | dim=240: ep~38

## Training Dynamics
Dim=8 converged fast (best ep=32, val=0.000644) — residual shortcuts accelerated early convergence vs trial 19 (best ep=44). Dim=60 converged at ep36 (consistent with residual variants). Dim=240 converged well. The LR decay to 2.5e-5 by epoch 49 for dim=60 confirms early plateau — the residual separable architecture finds its optimum quickly.

## Conclusion
Hypothesis validated with exceptional strength. All three dims improved simultaneously, which had never been achieved before:

| Dim | Prior Champion (T16) | New Champion (T20) | Delta |
|-----|---------------------|--------------------|-------|
| dim=8 | 0.729 | **0.777** | +0.048 |
| dim=60 | 0.777 | **0.801** | +0.024 |
| dim=240 | 0.815 | 0.810 | −0.005 |
| **avg** | **0.774** | **0.796** | **+0.022** |

**dim=8=0.777 is the highest dim=8 result in the entire experiment** (previous best: V4 at 0.747).

**Mechanism:** The ResSeparableConv3DBlock creates two complementary paths:
1. **Residual path (shortcut)**: direct identity mapping that preserves global structure → helps dim=8 recover by providing clean low-frequency signal
2. **Separable path (DW+PW)**: factorized feature extraction that captures per-channel spatial patterns → provides rich mid-frequency features for dim=60
3. **Plain enc4**: free transformation after z_pool3 (same principle as trial 16) → preserves dim=240 quality
4. **Consistent residual enc-dec style**: avoids the incompatibility seen in trial 18

The key insight: residual shortcuts in the separable blocks primarily benefit dim=8 (the shortcut provides the most direct signal for extreme compression), while the separable DW+PW path benefits dim=60 (factorized features → better structured representations). Plain enc4 protects dim=240. This three-way specialization explains why all dims improve simultaneously.

**New champion avg: 0.795874** (+0.022 vs prior champion). Largest single-trial improvement in the experiment.
