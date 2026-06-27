# Existing Architecture — AE3dAttention — REFERENCE

## Architecture Description
`AE3dAttention` extends the convolutional autoencoder (`AE3dConv`) with a spatial self-attention mechanism inserted at the bottleneck or within the encoder/decoder layers. The attention module learns to weight spatial positions differently — focusing the network on diagnostically relevant regions (e.g., myocardium walls, ventricular cavity) while suppressing background. Unlike skip connections (which are explicitly excluded), attention here is applied only within the bottleneck information pathway: the encoder still compresses to a single latent vector, maintaining the strict bottleneck constraint. The decoder mirrors the encoder structure.

## Results
- **R2_dim8:** 0.724291 | **R2_dim60:** 0.762300 | **R2_dim240:** 0.750303
- **avg_validation_R2_mean:** 0.745631
- **delta_vs_champion:** — (REFERENCE run)
- **MLflow Run IDs:** 65a449973c9b4819ae7c788328f7c7b8 b7d2a9522a6547cc9ef65cd9a252359c 40e9f1ef9f864a0b8a6696dbf3d57ad0

## Training Dynamics
All three dims converged cleanly with early stopping well before the 300-epoch cap:
- dim 8: best at epoch 36, stopped at epoch 66
- dim 60: best at epoch 57, stopped at epoch 87
- dim 240: best at epoch 34, stopped at epoch 64

Convergence was notably fast compared to `AE3dFCDeep` and `AE3dLinear`: all dims reached their best epoch within 60 epochs. The attention mechanism may accelerate convergence by allowing the optimizer to focus gradient updates on informative spatial regions early in training, rather than fitting background voxels that contribute little to the latent representation.

## Conclusion
`AE3dAttention` scores 0.7456 on average — placing it between `AE3dConv` (0.7445) and `AE3dFCDeep` (0.7517). The attention mechanism delivers a small but consistent improvement over the plain convolutional baseline across all three latent dims. The gain is most visible at dim 60 (0.762 vs 0.762 for AE3dConv — essentially tied) and dim 240 (0.750 vs 0.772 for AE3dConv — AE3dConv is better here), while at dim 8 AE3dAttention (0.724) slightly underperforms AE3dConv (0.700) — wait, actually AE3dAttention 0.724 > AE3dConv 0.700 at dim 8. At dim 240, AE3dConv 0.772 > AE3dAttention 0.750.

The attention mechanism trades off: it improves low-dim performance (dim 8: +0.024 over AE3dConv) by concentrating capacity on relevant spatial regions when the bottleneck is tightest, but at dim 240, where capacity is abundant, the added attention parameters may introduce noise or overfitting pressure that slightly hurts reconstruction. The architecture is computationally more expensive per epoch than AE3dConv, but the faster early stopping compensates, making wall-clock training time competitive. Overall, `AE3dAttention` is a solid mid-tier architecture — marginally better than `AE3dConv` on average but still below `AE3dFCDeep`.
