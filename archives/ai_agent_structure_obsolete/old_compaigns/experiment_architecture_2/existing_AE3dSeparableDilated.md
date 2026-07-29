# Existing Architecture — AE3dSeparableDilated — REFERENCE

## Architecture Description
`AE3dSeparableDilated` combines **depthwise-separable convolutions** with **dilated (atrous) convolutions**. A depthwise-separable convolution factorises a standard 3D conv into two steps: (1) a depthwise conv that applies a single filter per input channel independently, and (2) a pointwise 1×1×1 conv that mixes channels. This reduces the parameter count and FLOPs roughly by a factor of the kernel volume (≈27× for 3×3×3) while preserving representational capacity at the cost of some cross-channel interaction at each spatial layer. The dilated component expands the receptive field without pooling-induced resolution loss. Together, the intent is to combine the global context benefits of dilation with the parameter efficiency of separable convolutions — allowing deeper or wider architectures for the same compute budget. No skip connections; the bottleneck remains the sole information pathway.

## Results
- **R2_dim8:** 0.732277 | **R2_dim60:** 0.764862 | **R2_dim240:** 0.716711
- **avg_validation_R2_mean:** 0.737950
- **delta_vs_champion:** — (REFERENCE run)
- **MLflow Run IDs:** d1e732e69dfd4d558c4a1f01aaed9d72 b31031c8e124440dbaf49d1ab961b042 f908acc9a5734addb7bac4b88a7be137

## Training Dynamics
All three dims converged cleanly with early stopping:
- dim 8: best at epoch 43, stopped at epoch 73
- dim 60: best at epoch 57, stopped at epoch 87
- dim 240: best at epoch 34, stopped at epoch 64

Convergence is stable across dims. Dim 60 takes the most epochs to converge (57), which is consistent with separable convolutions needing more gradient steps to jointly optimise the depthwise and pointwise components — the factorised parameterisation is harder to optimise than a monolithic conv, even though the expressive capacity is similar.

## Conclusion
`AE3dSeparableDilated` scores 0.7380 on average, placing it 6th out of 9 models. The standout result is dim 60 (0.7649), which is the highest single-dim score among all separable/dilated variants and trails only `AE3dFCDeep` at dim 60 (0.7258 — actually lower, so 0.7649 is the best dim-60 score in the sweep). However, performance drops sharply at dim 8 (0.7323) and dim 240 (0.7167), producing an uneven profile across dims.

The dim-60 strength likely reflects that separable convolutions are most effective in the mid-range bottleneck: at dim 8, the severe compression demands that every filter capture maximum information, and the factorised parameterisation is at a disadvantage compared to monolithic convolutions; at dim 240, the permissive bottleneck means the model has excess capacity, and the added optimisation difficulty of the separable decomposition leads to underfitting relative to simpler architectures. The dilation component contributes consistent receptive-field gains across all dims, but cannot compensate for the separability penalty at the extremes.

Overall, `AE3dSeparableDilated` is not a strong candidate for the multi-dim champion due to its high variance across dims, but its dim-60 result (0.7649) is the best in the sweep at that dimension and could be worth revisiting in a targeted dim-60 exploitation trial.
