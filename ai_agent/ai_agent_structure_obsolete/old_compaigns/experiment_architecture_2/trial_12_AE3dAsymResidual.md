# Trial 12 — AE3dAsymResidual — CANDIDATE

## Hypothesis
Trial 11 (AE3dResidual) showed residual blocks are strong at dim=240 (0.7924) but weak at dim=8 (0.7276) due to isotropic pooling discarding z-information. The current champion V4 uses anisotropic pooling (pool1=(1,2,2), z_pool3=(2,1,1)) which is critical at low dims. Combining both mechanisms — V4's anisotropic z-pooling with residual encoder/decoder blocks — should produce stronger avg than either alone. Prediction: avg > 0.766648.

## Implementation
- Encoder: `ResConv3DBlock` replaces `Conv3DBlock` at all 4 stages. Shortcut is 1×1×1 conv when channels change (1→8, 8→16, 16→32, 32→64), Identity otherwise. MaxPool3d is external to the residual path.
- Pooling: identical to V4 — `pool1=MaxPool3d(1,2,2)` after enc1, `z_pool3=MaxPool3d(2,1,1)` between enc3/enc4, isotropic MaxPool inside enc2/enc3/enc4.
- Decoder: `ResUpConv3DBlock` replaces `UpConv3DBlock` at dec1/dec2/dec3. `ResConv3DBlock(downsample=False)` at dec4. Upsample layers unchanged.
- Bottleneck/FC: identical to V4 (bottleneck_conv 64→128→128, final_down Conv3d(k=2), FC).
- ~1.57M params at dim=8 (nearly identical to V4's 1.56M).

## Results
- **R2_dim8:** 0.739899 | **R2_dim60:** 0.798781 | **R2_dim240:** 0.747495
- **avg_validation_R2_mean:** 0.762058
- **delta_vs_champion** (trial avg − champion avg): −0.004590
- **MLflow Run IDs:** 17dcc0c19a3142139ff026c429e87848 f5173e0c03fc46d880b9c5364e9423e4 1f47e77309a1497db4a806b8b64baace
- **Best epochs:** 23/53 | 34/64 | ~22/52

## Training Dynamics
Consistent early stopping across all dims (ep 53, 64, ~52). Stable convergence with no spikes. LR decay to 6.25e-6 at dim=240, indicating the model plateaued early. dim=60 converged best: val loss 0.000576 vs V4's (unreported but implied similar). dim=240 notably weaker than trial 11's dim=240 (0.747 vs 0.792) — asymmetric pooling appears to hurt large-dim performance relative to isotropic pooling.

## Conclusion
The hypothesis partially held. Residual blocks + anisotropic pooling produced the best single-dim result in the experiment (dim=60: 0.7988, above champion+0.03 threshold → CANDIDATE). However, the avg fell short of the champion because dim=240 regressed significantly relative to trial 11's isotropic version (0.7475 vs 0.7924). The asymmetric z-pooling (which is beneficial for low-dim compression) appears to limit high-capacity representations at dim=240 — at that capacity, the model could benefit from richer spatial structure that isotropic pooling preserves. The trade-off between z-preservation (for dim=8) and spatial richness (for dim=240) is the key tension to resolve in future trials.
