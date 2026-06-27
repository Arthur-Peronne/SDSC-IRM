# Existing Architecture — AE3dDilated — REFERENCE

## Architecture Description
`AE3dDilated` is a convolutional autoencoder where the standard conv layers are replaced (at least in part) with **dilated (atrous) convolutions**. Dilated convolutions expand the receptive field exponentially without increasing the number of parameters or reducing spatial resolution through pooling — a dilation rate of d inserts d−1 zeros between filter taps, so a 3×3×3 kernel with dilation 2 covers a 5×5×5 region. Applied to 3D cardiac MRI (32×128×128 after ROI cropping), dilation allows the encoder to capture long-range spatial dependencies (e.g., the relationship between the left ventricle and the surrounding myocardium) at early layers, before spatial compression. No skip connections; the bottleneck remains the sole information pathway.

## Results
- **R2_dim8:** 0.749783 | **R2_dim60:** 0.738644 | **R2_dim240:** 0.756055
- **avg_validation_R2_mean:** 0.748160
- **delta_vs_champion:** — (REFERENCE run)
- **MLflow Run IDs:** 0e338713893f49f09e6e9efbd595512f 72d1c817ae5d4cd8937f080828956f8e d5350a5061ed4c948f9ede2ff70c1c7c

## Training Dynamics
All three dims converged cleanly with early stopping:
- dim 8: best at epoch 49, stopped at epoch 79
- dim 60: best at epoch 36, stopped at epoch 66
- dim 240: best at epoch 58, stopped at epoch 88

Convergence was consistent across dims, with best epochs clustering in the 36–58 range — notably faster than `AE3dFCDeep` at dim 8 (which typically needs more epochs to learn long-range FC structure) but comparable to the other convolutional models. No pathological behaviour; the LR scheduler was not needed.

## Conclusion
`AE3dDilated` scores 0.7482 on average, placing it 3rd in the sweep (behind AE3dFCDeep at 0.7517 and AE3dAttention at 0.7456). The dilated convolutions deliver a meaningful improvement over plain `AE3dConv` (0.7445) at dim 8 (+0.050) but the gains are more modest at dim 60 (−0.023, dilated slightly underperforms) and dim 240 (−0.016). This pattern is consistent with the hypothesis that expanded receptive fields are most valuable when the bottleneck is tightest: at dim 8, capturing global cardiac structure (whole-heart shape, ventricular ratio) without losing spatial precision is critical, and dilation achieves this efficiently. At dim 60 and 240, the bottleneck is large enough that standard convolutions can already assemble multi-scale features through depth, so the extra receptive field from dilation provides diminishing returns while potentially introducing redundancy. Overall, `AE3dDilated` is a strong low-dim architecture; its advantage over `AE3dConv` is most pronounced when the latent space is small.
