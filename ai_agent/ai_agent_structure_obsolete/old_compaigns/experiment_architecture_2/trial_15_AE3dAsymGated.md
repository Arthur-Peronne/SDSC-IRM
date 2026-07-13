# Trial 15 — AE3dAsymGated — FAILURE

## Hypothesis
Exploration (cooldown after trials 13+14 exploitation failures). Standard encoder conv blocks process features uniformly — all learned features are always applied. A gating mechanism would allow the encoder to selectively suppress or amplify feature channels at each stage, potentially adapting its representation strategy to preserve task-relevant spatial structure. The gate (1×1×1 sigmoid projection from input) acts as a learned channel-wise mask on the feature path output. V4's anisotropic pooling (pool1=(1,2,2), z_pool3=(2,1,1)) is preserved; decoder and bottleneck/FC are identical to V4. Prediction: adaptive feature selection would improve representations across all latent dims.

## Implementation
- New `GatedConv3DBlock`: feature path (Conv3d→IN→ReLU→Conv3d→IN) gated by a 1×1×1 conv from input (Sigmoid→IN). Output = ReLU(features * gate). MaxPool applied after gating.
- enc1–enc4 all use `GatedConv3DBlock` with the same channel progression as V4 (1→8→16→32→64).
- Pooling identical to V4: pool1=MaxPool3d(1,2,2) after enc1, z_pool3=MaxPool3d(2,1,1) between enc3/enc4, isotropic MaxPool inside enc2/enc3/enc4.
- Decoder: standard `UpConv3DBlock` (identical to V4).
- Bottleneck/FC: identical to V4 (bottleneck_conv 64→128→128, final_down Conv3d(k=2), FC, flattened_size=2048).
- ~1.57M params at dim=8 (nearly identical to V4 — gate adds only ~3K params via 1×1×1 convs).

## Results
- **R2_dim8:** 0.709698 | **R2_dim60:** 0.764387 | **R2_dim240:** 0.759303
- **avg_validation_R2_mean:** 0.744463
- **delta_vs_champion** (trial avg − champion avg): −0.022185
- **MLflow Run IDs:** 9020a9b263ab4d9d813760715fe64792 90cc90e712664bf38f5ed91373aa1df2 45fd4392c386422994b063747fd2b199
- **Best epochs:** 44/74 | 46/76 | 40/70

## Training Dynamics
All dims converged with early stopping at epochs 70–76. Best epochs at 40–46 (earlier than typical ~50+ in V4/residual trials), suggesting the gate mechanism causes earlier plateauing. Val losses at best epoch: dim=8=0.000767, dim=60=0.000637, dim=240=0.000665. The val losses are higher than trial 12 (dim=60 best=0.000576) despite similar param counts.

## Conclusion
The hypothesis failed. Gated convolutions degraded performance across all dims vs V4 (dim=8: 0.710 vs 0.747; dim=60: 0.764 vs 0.775; dim=240: 0.759 vs 0.778). The most likely mechanisms:

1. **Sigmoid saturation disrupts gradient flow.** The 1×1×1 gate applies Sigmoid, which saturates near 0 or 1. When a gate saturates near 0, the corresponding feature channel receives near-zero gradient, effectively killing that learning path. Unlike residual shortcuts (which guarantee a gradient highway via addition) or V4's plain convs, gating can create dead units during early training.

2. **Gate operates on unprocessed input, providing a weak signal.** The gate's 1×1×1 conv operates on the raw input (unfiltered) before the feature path has done any spatial processing. At enc1 (1→8 channels), the input is the raw MRI voxel values — a 1×1×1 projection provides almost no spatial context for making meaningful gating decisions. The gate signal is therefore noisy and unhelpful, forcing the feature path to learn despite unreliable channel modulation.

3. **Multiplicative interactions increase optimization difficulty.** For small latent dims (8), the FC must project from 2048 features to just 8 dimensions. The feature path × gate multiplication changes the effective learning landscape — instead of optimizing a linear+nonlinear chain, the network must jointly optimize multiplicative factors, which is harder with limited data (100 patients).

4. **No improvement from the gate vs. V4's plain convs.** V4's plain Conv3DBlocks are already capable of learning which features to emphasize (via learned weights). The gate adds a redundant selection layer that competes with, rather than complements, the existing channel-wise selectivity of the conv filters themselves.

**Key finding:** Multiplicative gating mechanisms are not beneficial for this reconstruction task with 100 training patients. The additive highway (residual, trial 12) is superior to the multiplicative highway (gated, trial 15) — residual addition preserves gradients unconditionally, while sigmoid gating conditionally blocks them.
