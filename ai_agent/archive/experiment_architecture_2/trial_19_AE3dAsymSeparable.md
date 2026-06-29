# Trial 19 — AE3dAsymSeparable — CANDIDATE

## Hypothesis
Trials 17 and 18 are 2 consecutive exploitation FAILUREs → cooldown, Exploration required. The reference separable model (AE3dSeparableDilated, REFERENCE) used depthwise separable convolutions with isotropic pooling and achieved avg=0.738. V4's anisotropic pooling (pool1=(1,2,2), z_pool3=(2,1,1)) was the primary driver of improvements in this experiment (+0.029 over isotropic variants). Combining depthwise separable convolutions with V4's anisotropic pooling has not been tried. Separable convolutions have fewer encoder-stage parameters (~13K vs ~150K for plain conv), which may reduce overfitting with 100 training patients. The hypothesis: separable + anisotropic pooling should substantially improve over the reference separable (avg 0.738 → ~0.767+).

## Implementation
- enc1: SeparableConv3DBlock(1, 8, downsample=False)
- pool1: MaxPool3d((1,2,2)) — anisotropic spatial downsampling, z preserved
- enc2: SeparableConv3DBlock(8, 16, downsample=True)
- enc3: SeparableConv3DBlock(16, 32, downsample=True)
- z_pool3: MaxPool3d((2,1,1)) — z halved mid-encoder, spatial preserved
- enc4: SeparableConv3DBlock(32, 64, downsample=True)
- Bottleneck/FC: identical to V4 (Conv3d 64→128→128, final_down stride-2, flattened_size=2048)
- Decoder: standard UpConv3DBlock (plain, not residual — safe choice for new encoder family)
- ~1.36M params at dim=8 (vs V4's 1.56M — lighter encoder stages)

## Results
- **R2_dim8:** 0.703101 | **R2_dim60:** 0.804499 | **R2_dim240:** 0.757966
- **avg_validation_R2_mean:** 0.755189
- **delta_vs_champion** (trial avg − champion avg): −0.018617
- **MLflow Run IDs:** 75a8b0a784c74631a29d73ed709d236b 76903887db9a46d8b27bc4a8797326d6 3b5ce5cc173d4100abb9aab1e6c6c78a
- **Best epochs:** dim=8: ep44 | dim=60: ep58 | dim=240: ep38

## Training Dynamics
Dim=8 converged early (best ep=44, val=0.000821) with LR fully decayed — consistent with limited gradient signal for low-capacity latent spaces using separable convolutions. Dim=60 converged well (best ep=58, val=0.000562) and produced the strongest dim=60 result in the entire experiment (0.804). Dim=240 converged earliest of the three (best ep=38, val=0.000686), which is unusually early for a high-capacity dim — suggests the separable encoder + plain decoder combination reaches its capacity limit earlier for dim=240 than residual variants.

## Conclusion
CANDIDATE. The hypothesis was partially validated. Separable + V4 pooling substantially improves over the reference separable (avg 0.738 → 0.755, +0.017). However, it does not surpass the current champion (avg 0.773806). The result profile is strongly biased: **dim=60=0.804 is the best dim=60 performance in the entire experiment** (surpassing champion's 0.777 by +0.027), while dim=8 is the weakest seen (0.703) and dim=240 is moderate (0.758, below champion's 0.815).

**Key finding:** Depthwise separable convolutions are exceptionally effective for mid-capacity latent spaces (dim=60). The per-channel spatial feature extraction (depthwise) followed by channel mixing (pointwise) naturally produces rich spatial features well-suited to 60-dimensional compression. For dim=8, the separable convolution's reduced expressiveness per-stage limits what the bottleneck can encode at extreme compression. For dim=240, the plain decoder becomes a bottleneck (vs residual decoder in the champion) — decoder expressiveness matters more at high capacity.

**Design insight:** The separable encoder is a strong fit for dim=60 optimization. A future trial combining separable enc1-enc3 with V4's enc4 (plain conv, post-z_pool3) and the residual decoder from trial 16 could potentially recover dim=240 while preserving the dim=60 strength — but enc-dec consistency must be respected (trial 18 showed mixing plain encoder + residual decoder can be catastrophic).

**Kept as CANDIDATE** for future analysis; dim=60=0.804 is the highest single-dim result for dim=60 in the experiment.
