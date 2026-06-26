# Trial 19 — AE3dDilatedAttentionSEDecoder2 — FAILURE

## Hypothesis
Trial 15 (SE on all 4 decoder blocks, reduction=4) missed the champion by only Δlb=-0.005. I hypothesized that applying SE to the two high-channel decoder blocks only (dec1: 128→64, dec2: 64→32) with a stronger reduction ratio (reduction=8 instead of 4) would concentrate attention where it matters most. The low-channel decoder blocks (dec3: 32→16, dec4: 16→8) have few channels and provide minimal benefit from SE attention; the small SE gates there risk introducing noise. The net result should be a lighter, more focused version of trial 15: fewer parameters (+1,280 over champion vs trial 15's +2,048), stronger per-block attention signal.

## Implementation
New `SEUpConv3DBlock(in_channels, out_channels, reduction=8)`: upconv + 2×(Conv3d+IN+ReLU) + SEBlock3D. Champion encoder and bottleneck unchanged. `dec1 = SEUpConv3DBlock(128, 64, reduction=8)`, `dec2 = SEUpConv3DBlock(64, 32, reduction=8)`, `dec3 = UpConv3DBlock(32, 16)`, `dec4 = UpConv3DBlock(16, 8)`.

## Results
- **validation_R2_mean:** 0.780743
- **validation_R2_std:** 0.130170
- **val_R2_lower_bound** (mean − std): 0.650573
- **lower_bound_compared_to_champion** (trial lb − champion lb): -0.080162
- **mean_compared_to_champion** (trial mean − champion mean): -0.022895
- **MLflow Run ID:** e71a7525146f4fa4ac0753caabef5067
- **Best epoch:** 57 / 87 (early stop)

## Training Dynamics
Early stopping at epoch 87 (best epoch 57). Validation std jumped to 0.130 (vs champion's 0.073 and trial 15's 0.077) — even worse variance than the full-decoder SE variant. Mean R2 also regressed (0.781 vs trial 15's 0.802). Training R2 (0.897) vs validation R2 (0.781) shows a clear overfitting gap.

## Conclusion
The hypothesis failed. The "focused SE on high-channel blocks only" approach performed worse than even the full-decoder SE of trial 15, which itself barely missed the champion. Two observations:

1. **Higher variance than trial 15.** Despite fewer SE parameters (+1,280 vs +2,048), std=0.130 is higher than trial 15's std=0.077. Concentrating SE on only two blocks may create an asymmetric decoder: dec1 and dec2 have learned channel re-weighting while dec3 and dec4 do not, breaking the smooth gradient flow from deep to shallow decoder stages.

2. **Decoder SE consistently fails to beat champion.** Both trial 15 (all 4 blocks) and this trial (2 blocks) failed. The champion's plain UpConv3DBlock decoder is already well-calibrated for this dataset size. Any SE addition, regardless of placement or intensity, increases variance beyond the gain in mean accuracy.

**Pattern emerging:** SE attention is effective in the encoder (champion uses it in all 4 encoder stages) but counter-productive in the decoder for 150-patient cardiac MRI data. The decoder reconstructs from a 120-dim latent — the bottleneck constraint already forces channel selection upstream. Adding channel attention downstream re-selects already-selected features, introducing unnecessary parameter overhead and instability.
