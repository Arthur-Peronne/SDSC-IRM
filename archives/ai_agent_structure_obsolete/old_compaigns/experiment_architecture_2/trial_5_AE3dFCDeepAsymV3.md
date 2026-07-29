# Trial 5 — AE3dFCDeepAsymV3 — FAILURE

## Hypothesis
Trial 4 (AE3dFCDeepAsymV2) showed that MaxPool3d(2,1,1) z-pooling helps dim=8 (+0.045 vs T3) but regresses dims 60/240 (−0.025, −0.010 vs T3). The report identified the mechanism: MaxPool hard-selects one z-slice's maximum activation and discards the other, losing reconstruction-relevant information for large dims.

The hypothesis: replacing MaxPool(2,1,1) with a learned Conv3d(128,128,kernel=(2,1,1)) + InstanceNorm3d + ReLU allows the network to learn optimal channel-specific weighted combinations of adjacent z-pairs. At dim=8, the learned conv could learn to approximate max-selection (recovering the dim=8 benefit); at dims 60/240, it could learn softer combinations that preserve more z-diversity for reconstruction.

## Implementation
All layers identical to AE3dFCDeepAsymV2 except:
- **z_pool**: `nn.Sequential(Conv3d(128,128,kernel=(2,1,1),stride=(2,1,1)), InstanceNorm3d(128), ReLU)` instead of MaxPool3d(2,1,1)
- **z_up**: `ConvTranspose3d(128,128,kernel=(2,1,1),stride=(2,1,1))` instead of Upsample(2,1,1)
- Total params at dim=8: 1,628,809 (+65,792 vs V2 due to z_pool conv and z_up convtranspose)

## Results
- **R2_dim8:** 0.716267 | **R2_dim60:** 0.688053 | **R2_dim240:** 0.759172
- **avg_validation_R2_mean:** 0.721164
- **delta_vs_champion** (trial avg − champion avg): −0.045299
- **MLflow Run IDs:** 0bafd21630c14424a99556adbf7081dd aea4914497bd45758cca7f94b67a2a9b 3d6ed17be70a4d03ad9ad0b2941f9ec8
- **Best epochs:** 54/300 | 30/300 | 47/300

## Training Dynamics
dim=60 converged fastest of the three runs (best epoch 30/300), with early stopping at epoch 60 — the shortest dim=60 run seen across all trials. This very early plateau, combined with the worst R2 (0.688), suggests the optimization landscape for the learned z-pooling is poorly conditioned at this intermediate latent dim. The training R2 at dim=60 (0.797) is far above validation R2 (0.688), indicating a generalization gap absent in V2 — the learned z-pooling may have overfit the training distribution's z-slice patterns. dim=240 had the most typical convergence profile (epoch 47 best) but still substantially underperformed relative to T3/T4.

## Conclusion
The hypothesis did not hold. Learned z-pooling performed worse than MaxPool across all three dims:

- **dim=8:** 0.716 vs V2-champion 0.748 (Δ=−0.031) — moderate regression
- **dim=60:** 0.688 vs V2-champion 0.760 (Δ=−0.072) — catastrophic regression
- **dim=240:** 0.759 vs V2-champion 0.792 (Δ=−0.033) — moderate regression

The dim=60 result (0.688) is the worst single-dim result across all trials, worse even than T1 (AE3dFCDeepDilated) and T2 (AE3dFCDeepSE).

Three mechanisms explain the failure:

1. **Optimization interference at dim=60:** The Conv3d(2,1,1) z-pooling layer is asked to simultaneously learn z-compression (combining two z-slices into one) and feature refinement (InstanceNorm + ReLU introduces a new non-linearity). This dual role is conflated in one layer, making it harder to optimize than MaxPool (which has a fixed, deterministic role). At dim=60, the intermediate compression ratio (2048→60, 34x) may create a gradient landscape where the z-pooling layer receives conflicting signals — sometimes pressured to compress z, sometimes to preserve spatial detail.

2. **ConvTranspose3d z-upsample instability:** The decoder's `z_up = ConvTranspose3d(128,128,kernel=(2,1,1))` must reconstruct 4 z-slices from 2. In V2, this was done by a parameter-free Upsample (trivially invertible). With ConvTranspose3d, the network must learn the inverse mapping, adding an additional optimization burden on the decoder. ConvTranspose layers are known to produce checkerboard artifacts; applied along z with stride 2 and kernel 2, the resulting decoded features may have z-aliasing.

3. **InstanceNorm after z-pool:** InstanceNorm3d after the Conv3d(2,1,1) normalizes each channel's spatial distribution. However, after z-pooling the feature map is 128×2×8×8 — only 2 z-positions remain. InstanceNorm over 2×8×8 = 128 elements per channel is much less statistically stable than over 4×8×8 = 256 elements. The normalization statistics are noisy, introducing instability in the feature map before final_down.

**Key takeaway:** For the z-pooling step specifically, MaxPool3d(2,1,1) is superior to a learned Conv3d(2,1,1) because: (a) MaxPool's role is unambiguous (select the peak activation), (b) it has no learnable parameters that compete with the rest of the network, and (c) its inverse (Upsample) is trivial. Future trials should treat z-pooling as a structural constraint (MaxPool) rather than a learned transformation.
