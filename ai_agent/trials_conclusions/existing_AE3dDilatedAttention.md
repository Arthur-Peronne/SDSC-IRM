# Existing Architecture — AE3dDilatedAttention — REFERENCE

## Architecture Description
`AE3dDilatedAttention` combines two previously explored strategies: **dilated convolutions** from `AE3dDilated` and **Squeeze-and-Excitation (SE) channel attention** from `AE3dAttention`. Dilated convolutions expand the receptive field without resolution loss, while SE blocks perform channel-wise feature recalibration — rescaling each feature map by a learned scalar that reflects its global importance (computed via global average pooling → two FC layers → sigmoid). The intent is synergistic: dilation captures long-range spatial context, and SE gates suppress uninformative channels at each scale, directing the bottleneck toward the most diagnostically relevant cardiac features. No skip connections; the bottleneck remains the sole information pathway.

## Results
- **R2_dim8:** 0.739938 | **R2_dim60:** 0.741982 | **R2_dim240:** 0.719198
- **avg_validation_R2_mean:** 0.733706
- **delta_vs_champion:** — (REFERENCE run)
- **MLflow Run IDs:** 2492702da6254a649b1c597ee382daf8 e6e3180bc5024dc2abcf8613db928641 3e03d6da5bc74c3bbac2c28a51a3d9ab

## Training Dynamics
All three dims converged cleanly with early stopping:
- dim 8: best at epoch 47, stopped at epoch 77
- dim 60: best at epoch 43, stopped at epoch 73
- dim 240: best at epoch 34, stopped at epoch 64

Convergence was stable across dims with no pathological behaviour. Best epochs (34–47) are somewhat earlier than `AE3dDilated` alone (36–58), suggesting that the SE gating helps the optimizer converge faster by concentrating gradient signal on relevant channels from early training.

## Conclusion
`AE3dDilatedAttention` scores 0.7337 on average, placing it 7th out of 8 evaluated models — notably below both `AE3dDilated` (0.7482) and `AE3dAttention` (0.7456) individually. The combination fails to deliver the expected synergy, and the gap is largest at dim 240 (0.7192 vs 0.7561 for `AE3dDilated`), where the added SE parameters appear to hurt rather than help.

This is consistent with the pattern observed in experiment_architecture_1, where `AE3dDilatedAttention` variants dominated the leaderboard at latdim=120 but that performance did not generalise to other latent dims. The likely mechanism: SE blocks add a multiplicative non-linearity on top of dilated feature maps. At low dims (8, 60), the bottleneck is already a hard constraint, and SE helps focus the limited capacity. At high dims (240), the bottleneck is permissive and the SE scaling introduces an additional optimisation difficulty — the model must simultaneously learn which channels are useful (SE) and how to use them (decoder), leading to worse final reconstruction compared to plain dilated convolutions that learn directly. The architectural hypothesis was valid for single-dim evaluations at intermediate latdim, but the multi-dim protocol reveals it is not robust across the latdim range.
