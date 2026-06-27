# Existing Architecture — AE3dFCDeep — REFERENCE

## Architecture Description
`AE3dFCDeep` is a fully-connected deep autoencoder operating on flattened 3D volumes. The encoder flattens the 1×32×128×128 input into a 1D vector and passes it through a stack of linear layers with progressive dimensionality reduction (e.g., 524288 → 4096 → 1024 → 256 → latent). The decoder mirrors this with linear layers expanding back to full resolution. No convolutions, no spatial pooling, no skip connections — pure bottleneck via matrix multiplications. The architecture relies entirely on global weight sharing across the volume rather than local receptive fields.

## Results
- **R2_dim8:** 0.771515 | **R2_dim60:** 0.725822 | **R2_dim240:** 0.757791
- **avg_validation_R2_mean:** 0.751709
- **delta_vs_champion:** — (REFERENCE run)
- **MLflow Run IDs:** 0dee39dc65294cb69d33f1c99ea60ce0 9d1b0b8b751a4208911bc5ac714702a2 5140426136f643b499b089003010c28a

## Training Dynamics
All three dims converged with early stopping:
- dim 8: best at epoch 68, stopped at epoch 98 (patience exhausted)
- dim 60: best at epoch 44, stopped at epoch 74
- dim 240: best at epoch 41, stopped at epoch 71

Dim 8 required the most epochs to converge, consistent with the higher difficulty of compressing to 8 dimensions — the loss surface is narrower and the optimizer takes longer to find the optimum. The LR scheduler fired multiple times across all runs (halving on plateau), with the final few epochs at very low LR (~3e-6) contributing negligible improvement.

## Conclusion
`AE3dFCDeep` outperforms `AE3dCurrent` on average (0.7517 vs 0.7132), particularly at dim 8 (0.7715 vs 0.7133). This is notable: the fully-connected architecture, despite operating on the flattened volume without any spatial inductive bias, achieves better compression at low latent dimensions. This suggests that the global weight sharing of FC layers can learn cross-spatial correlations (e.g., long-range anatomical dependencies across the ventricle) that local convolutional filters miss at very small latent codes. At dim 60, performance drops below `AE3dCurrent` (0.7258 vs 0.7330), likely because FC layers overparameterize the intermediate representations when the latent code is large enough — the network memorizes rather than generalizes spatial structure. At dim 240, the FC approach recovers (0.7578 vs 0.6934), again showing that the FC architecture handles high-capacity bottlenecks more gracefully than conv.
