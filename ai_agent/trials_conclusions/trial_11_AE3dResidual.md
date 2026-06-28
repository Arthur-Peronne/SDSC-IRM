# Trial 11 — AE3dResidual — FAILURE

## Hypothesis
Replacing standard Conv3DBlocks with residual blocks (output = ReLU(F(x) + shortcut(x))) in both encoder and decoder would improve gradient flow and allow the network to learn identity + correction jointly. This should produce better bottleneck representations compared to AE3dFCDeep, which uses plain conv stacks. The shortcut is a 1×1×1 projection conv when channels change, Identity otherwise. MaxPool downsampling stays external to the residual path. Expected improvement over champion AE3dFCDeepAsymV4 (avg=0.766648).

## Implementation
- New `ResConv3DBlock`: two Conv3d(3×3×3) with IN+ReLU, shortcut via Conv3d(1×1×1) when in≠out channels, MaxPool3d after the residual sum.
- New `ResUpConv3DBlock`: ConvTranspose3d upsampling then residual conv block with Identity shortcut (channels already match post-upconv).
- Overall topology identical to AE3dFCDeep: 4 encoder stages (1→8→16→32→64), bottleneck_conv (64→128→128), final_down Conv3d(k=2,s=2), FC, mirror decoder.
- ~1.57M params at dim=8 (vs ~1.56M for AE3dFCDeepAsymV4) — parameter count nearly identical; difference is only the 1×1×1 shortcut convs.

## Results
- **R2_dim8:** 0.727587 | **R2_dim60:** 0.776836 | **R2_dim240:** 0.792391
- **avg_validation_R2_mean:** 0.765604
- **delta_vs_champion** (trial avg − champion avg): −0.001044
- **MLflow Run IDs:** def7a5ec255146a4b8f37bd5e7a1dea0 30d45a10fb3c48f7b4260b28d6ebaf6a 12761a970c454f3fa9724b8c3abaf302
- **Best epochs:** 33/63 | 32/62 | ~46/76 (estimated from early-stop pattern)

## Training Dynamics
Consistent early stopping across all three dims (ep 63, 62, ~76), mirroring the AE3dFCDeep family pattern. Convergence was stable with no spikes. The LR scheduler stepped down normally. No dim showed pathological behaviour. The dim=240 result (0.7924) is the strongest single-dim score seen so far in the current experiment, suggesting residual learning is beneficial at higher latent capacities.

## Conclusion
The hypothesis partially held: residual learning improves reconstruction quality at dim=240 (best single-dim result in the experiment). However, the isotropic pooling structure (all four stages use MaxPool3d(2,2,2)) limits performance at dim=8 — the champion AE3dFCDeepAsymV4 uses anisotropic pooling (MaxPool3d(1,2,2) at stage 1 + z_pool between enc3/enc4) which better preserves the z-dimension's limited depth (32 slices). The residual mechanism did not compensate for this spatial information loss at low dims. The combination of anisotropic pooling (from V4) with residual blocks is a natural next step if exploration is exhausted and exploitation resumes.
