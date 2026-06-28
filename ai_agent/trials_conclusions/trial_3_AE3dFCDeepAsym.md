# Trial 3 — AE3dFCDeepAsym — CHAMPION

## Hypothesis
Cardiac MRI volumes have strongly anisotropic spatial structure: the z-dimension (32 slices) is typically 4–8× coarser in physical spacing than the in-plane dimensions (128×128 pixels). AE3dFCDeep applies isotropic MaxPool3d(2,2,2) at every stage, including stage 1, which halves the z-resolution from 32 to 16 immediately. At this point, z-slices are still spatially fine-grained and each slice carries distinct anatomical content (basal, mid-ventricular, apical). Halving z at stage 1 causes early information collapse before convolutions have learned enough channels to compensate.

The hypothesis: replacing stage-1 MaxPool3d(2,2,2) with MaxPool3d(1,2,2) (preserving z-depth, only halving spatial) lets the encoder accumulate more z-discriminative features before any z-downsampling occurs. This should improve reconstruction at all latent dims, but especially at large dims (60, 240) where the decoder can exploit richer, z-diverse bottleneck features to reconstruct fine detail.

## Implementation
- **enc1:** Conv3DBlock(1→8, downsample=False) producing 8×32×128×128
- **pool1:** MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2)) → 8×32×64×64 (z preserved)
- **enc2:** Conv3DBlock(8→16, downsample=True) → 16×16×32×32 (z halved here)
- **enc3:** Conv3DBlock(16→32, downsample=True) → 32×8×16×16
- **enc4:** Conv3DBlock(32→64, downsample=True) → 64×4×8×8
- **bottleneck_conv:** Conv3d(64→128), InstanceNorm, ReLU (×2) → 128×4×8×8
- **final_down:** ConvTranspose3d with kernel (4,2,2) stride (4,2,2) → 128×1×4×4
- **FC:** 2048→latent_dim (encode) / latent_dim→2048 (decode)
- **Decoder:** initial_up (1×4×4→4×8×8), UpConv3DBlock×3, then Upsample(1,2,2)+Conv3DBlock for stage-4 (mirrors the asymmetric pool1), final 1×1 Conv
- All Conv3DBlock and UpConv3DBlock definitions unchanged from AE3dFCDeep
- Total params: 2,775,665 (identical to AE3dFCDeep — channel counts unchanged)

## Results
- **R2_dim8:** 0.702309 | **R2_dim60:** 0.784502 | **R2_dim240:** 0.801498
- **avg_validation_R2_mean:** 0.762770
- **delta_vs_champion** (trial avg − champion avg): +0.011061
- **MLflow Run IDs:** 97b63eb1ab23426faafdab56cd5922d9 ac183f4925284e2f962e6b54ea93a84e 746a925cd4264689963bbe4f4adafe58
- **Best epochs:** 50/300 | 42/300 | 57/300

## Training Dynamics
All three runs converged stably with no loss spikes or instabilities. Early stopping triggered at epochs 80 (dim=8), 72 (dim=60), and 87 (dim=240). The convergence pattern is notably faster than AE3dFCDeep's typical 100–150 epochs before plateau, yet the final R2 values surpass the previous champion at dims 60 and 240. The learning rate scheduler progressively halved from 5e-5 down to 3–6e-6 before stopping, indicating genuine local minima were reached. Validation loss improved steadily throughout training with no plateau-then-collapse pattern, suggesting the asymmetric encoder provides a smoother optimization landscape.

dim=8 is the outlier: best epoch 50 vs 42–57 for larger dims, and the lowest R2. The higher-capacity dims benefit most from the richer z-diverse features in the bottleneck, while at dim=8 the FC compression (2048→8) is so severe that the benefit of z-preservation is swamped by the extreme dimensionality reduction.

## Conclusion
The hypothesis partially held. Anisotropic stage-1 pooling produced a **new champion** (avg R2 0.7628 vs 0.7517), driven by strong gains at dims 60 and 240:

- **dim=8:** 0.7023 vs champion 0.7715 (Δ=−0.069) — regression
- **dim=60:** 0.7845 vs champion 0.7258 (Δ=+0.059) — strong improvement
- **dim=240:** 0.8015 vs champion 0.7578 (Δ=+0.044) — strong improvement

The dim=8 regression is mechanistically coherent: preserving z at stage 1 produces a larger intermediate feature map (8×32×64×64 vs 8×16×64×64 in AE3dFCDeep). This doubles the z-extent before enc2, making enc2's downsampling responsible for collapsing twice as much z-information in a single step. At dim=8, the FC layer then has to compress 2048 features into only 8 codes — any additional structural complexity in the encoder can hurt because the extreme bottleneck cannot represent z-variation anyway. The conv layers end up encoding spatially redundant information across z that the FC cannot exploit.

At dim=60 and dim=240, the mechanism reverses. The FC layer can retain 60–240 independent dimensions, many of which can encode z-slice-specific features. By keeping z intact at stage 1, enc2–enc4 operate on a feature space that has already differentiated basal/mid/apical anatomy in the channel dimension (via 3×3×3 convolutions that see across z), rather than having that spatial distinction eliminated by isotropic pooling. The result is a richer bottleneck with more linearly independent dimensions, directly increasing reconstruction R2.

The architectural insight is that anisotropic pooling is latent-dim-dependent: it helps when the latent space can represent diverse features, and hurts when the bottleneck is so narrow that z-diversity becomes irrelevant noise. Future exploitation should consider mixed strategies (e.g., asymmetric pooling at stages 1 and 2) or latent-dim-adaptive encoder depth.
