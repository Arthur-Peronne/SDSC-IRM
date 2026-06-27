# Existing Architecture — AE3dConv — REFERENCE

## Architecture Description
`AE3dConv` is a pure convolutional 3D autoencoder without any fully-connected layers in the bottleneck pathway. The encoder applies a series of 3D convolutions with stride or max-pooling to progressively downsample the spatial volume, maintaining channel depth growth throughout. The latent representation is a spatially-structured feature map that is directly decoded by transposed convolutions. No linear projection to a 1D latent vector, no skip connections — the bottleneck capacity is controlled entirely by the channel count and spatial resolution at the deepest layer.

## Results
- **R2_dim8:** 0.700152 | **R2_dim60:** 0.761664 | **R2_dim240:** 0.771566
- **avg_validation_R2_mean:** 0.744460
- **delta_vs_champion:** — (REFERENCE run)
- **MLflow Run IDs:** b2733e91203f45558850e66a7f89bab3 2db3eddbff364a348dbee128173bd591 17b3168b5bb3439f80bf976234d9bc33

## Training Dynamics
All three dims converged cleanly with early stopping:
- dim 8: best at epoch 44, stopped at epoch 74
- dim 60: best at epoch 41, stopped at epoch 71
- dim 240: best at epoch 50, stopped at epoch 80

Dim 240 required the most epochs to reach its best, consistent with a larger bottleneck taking longer to fill and regularize. The LR scheduler fired multiple times across all runs; no instability or loss spikes observed.

## Conclusion
`AE3dConv` scores 0.7445 on average, placing it third behind `AE3dFCDeep` (0.7517) and `AE3dCurrent` (0.7132) but ahead of `AE3dCurrent`. The most striking pattern is the strong monotonic increase with latent dim: dim 8 (0.700) → dim 60 (0.762) → dim 240 (0.772). This is the opposite of `AE3dCurrent`'s profile where dim 240 underperformed dim 60. The spatially-structured bottleneck of `AE3dConv` scales well with capacity — more channels means more spatial detail is preserved, and the convolutional decoder can exploit this directly without the information bottleneck imposed by a linear projection. The weak dim 8 performance (0.700 vs 0.7715 for `AE3dFCDeep`) suggests that a purely spatial latent code is an inefficient basis at very low dimensionality — a 1D projection via FC layers can learn a more compressed global representation than a spatially-localized feature map with few channels.
