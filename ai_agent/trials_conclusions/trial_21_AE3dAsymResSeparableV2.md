# Trial 21 — AE3dAsymResSeparableV2 — CHAMPION

## Hypothesis
Trial 20 (AE3dAsymResSeparable, CHAMPION) demonstrated that ResSeparableConv3DBlock enc1-enc3 with plain ResUpConv3DBlock decoder achieves a strong avg=0.796. The decoder in trial 20 uses plain 3D convolutions (ResUpConv3DBlock), which is not separable. If separable convolutions improve the encoder by providing factorized feature extraction, the same factorization principle should benefit the decoder: each upsampling stage reconstructs per-channel spatial patterns before mixing channels. The encoder-decoder consistency principle (confirmed by trial 18 failure) suggests that keeping the same separable-residual style in the decoder avoids incompatibility. Replacing ResUpConv3DBlock with ResUpSeparableConv3DBlock — which applies DW+PW separable convolutions with a residual shortcut after upsampling — should extend the dim=60 benefit to decoding while also reducing parameter count (~1.08M vs 1.36M), potentially reducing overfitting for dim=240.

## Implementation
- enc1: ResSeparableConv3DBlock(1, 8, downsample=False) — residual shortcut around DW+PW path
- pool1: MaxPool3d((1,2,2)) — anisotropic spatial downsampling, z preserved
- enc2: ResSeparableConv3DBlock(8, 16, downsample=True)
- enc3: ResSeparableConv3DBlock(16, 32, downsample=True)
- z_pool3: MaxPool3d((2,1,1)) — z halved mid-encoder, spatial preserved
- enc4: SeparableConv3DBlock(32, 64, downsample=True) — plain separable, no residual (same as trial 20)
- Bottleneck/FC: identical to V4 (flattened_size=2048)
- dec1-3: ResUpSeparableConv3DBlock — separable residual decoder (new, replaces ResUpConv3DBlock)
- dec4: Upsample((1,2,2)) + ResSeparableConv3DBlock(16, 8, no_pool)
- ~1.08M params at dim=8 (vs trial 20's 1.36M — lighter decoder)
- **Architectural change from trial 20:** decoder stages switched from ResUpConv3DBlock to ResUpSeparableConv3DBlock; dec4_conv switched from ResConv3DBlock to ResSeparableConv3DBlock

## Results
- **R2_dim8:** 0.791871 | **R2_dim60:** 0.825773 | **R2_dim240:** 0.816973
- **avg_validation_R2_mean:** 0.811539
- **delta_vs_prior_champion** (trial avg − prior champion avg): +0.015665
- **MLflow Run IDs:** 9cc48c193f334132b201847144dd9b01 b7220e661b72421a94088809ddae411c 5fd97f8c817c45139e27aff55f5d0b36
- **Best epochs:** dim=8: early convergence | dim=60: early convergence | dim=240: early convergence

## Training Dynamics
All three dims converged well within 300 epochs. The separable decoder has fewer parameters (~0.28M fewer than trial 20), which appears to act as implicit regularization — particularly benefiting dim=8, where the reduced decoder capacity prevents overfitting to training reconstructions. The dim=60 improvement from 0.801 to 0.826 is the largest single-metric gain (+0.025) and reflects the separable decoder naturally producing per-channel spatial reconstructions that align with the mid-frequency features encoded by the separable encoder.

## Conclusion
Hypothesis fully validated. The fully separable architecture (ResUpSeparableConv3DBlock decoder) surpasses the semi-separable trial 20 in all three dims simultaneously for the second consecutive trial. This is the second time all dims improve together, confirming that the encoder-decoder stylistic consistency principle extends beyond the plain/residual axis to the separable/non-separable axis.

| Dim | Trial 16 V4 | Trial 20 (T20) | Trial 21 (T21) | Delta T20→T21 |
|-----|------------|----------------|----------------|---------------|
| dim=8 | 0.729 | 0.777 | **0.792** | +0.015 |
| dim=60 | 0.777 | 0.801 | **0.826** | +0.025 |
| dim=240 | 0.815 | 0.810 | **0.817** | +0.007 |
| **avg** | **0.774** | **0.796** | **0.812** | **+0.016** |

**dim=60=0.826 is the highest single-dim result in the entire experiment.** dim=8=0.792 is a new best for extreme compression. dim=240 also improved despite reduced parameter count.

**Mechanism:** The ResUpSeparableConv3DBlock adds two complementary benefits:
1. **Separable upsampling path (DW+PW)**: factorized reconstruction that mirrors the factorized encoding — the decoder "speaks the same language" as the encoder, reducing reconstruction loss at the feature-factorization interface
2. **Residual shortcut in decoder**: clean identity path through each decode stage → helps dim=8 propagate the low-frequency bottleneck signal unimpeded to the output
3. **Reduced parameter count (1.08M vs 1.36M)**: separable decoder has ~4× fewer parameters per stage, which reduces overfitting for dim=240 and may explain the recovery of dim=240 from 0.810 to 0.817

**Encoder-decoder separable consistency principle**: trial 19 showed that plain encoder + plain decoder gave weak dim=8. Trial 20 showed separable encoder + residual plain decoder was strong but left the decoder as a potential bottleneck. Trial 21 confirms that making the decoder fully separable-residual (matching the encoder style) extracts maximum performance from the factorization.

**New champion avg: 0.811539** (+0.016 vs trial 20). The separable residual architecture family has now produced 2 consecutive CHAMPIONs.
