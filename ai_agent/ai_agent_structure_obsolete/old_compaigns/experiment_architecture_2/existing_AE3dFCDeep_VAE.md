# Existing Architecture — AE3dFCDeep_VAE — REFERENCE

## Architecture Description
`AE3dFCDeep_VAE` is the variational autoencoder version of `AE3dFCDeep`. The encoder is identical — a stack of fully-connected linear layers projecting from the flattened volume to a latent space — but the latent layer outputs both a mean (μ) and log-variance (log σ²) vector. The reparameterization trick samples z = μ + σ·ε (ε ~ N(0,I)) during training. The decoder is also identical to AE3dFCDeep. The loss function combines reconstruction MSE with a KL divergence term (β·KL), with β ramping linearly from 0 to 0.00022 over the first 41 epochs (β-VAE warmup). No skip connections.

## Results
- **R2_dim8:** 0.728898 | **R2_dim60:** 0.770488 | **R2_dim240:** 0.724632
- **avg_validation_R2_mean:** 0.741339
- **delta_vs_champion:** — (REFERENCE run)
- **MLflow Run IDs:** ab5b8c351c8c4fb5a8c554fafb284ab2 8449767e1e684d51a1e8e63967a6cf8a 3f1d8794d0994438a9b38ea6f4281d19

## Training Dynamics
All three dims converged cleanly with early stopping:
- dim 8: best at epoch 47, stopped at epoch 77
- dim 60: best at epoch 63, stopped at epoch 93
- dim 240: best at epoch 76, stopped at epoch 106

The KL warmup (β ramping over 41 epochs) is visible in the training dynamics: early epochs are dominated by reconstruction loss while the KL term has negligible weight, allowing the network to first learn a good reconstruction basis before the regularization pressure pushes the latent space toward N(0,I). This produces more stable early convergence than a fixed β.

## Conclusion
`AE3dFCDeep_VAE` scores 0.7413 on average — lower than its deterministic counterpart `AE3dFCDeep` (0.7517) but competitive. The VAE regularization imposes a trade-off: the KL term forces the latent distribution toward a unit Gaussian, which reduces reconstruction fidelity but improves latent space regularity (interpolability, generative quality). The gap is most pronounced at dim 240 (0.725 vs 0.758), where the KL penalty is proportionally more constraining — 240 independent Gaussian constraints vs 8. Interestingly, at dim 60 the VAE nearly matches AE3dFCDeep (0.770 vs 0.726), suggesting that at intermediate latent sizes the regularization pressure is beneficial — it prevents the encoder from encoding noise and forces the network to learn a more structured representation. The VAE is a strong candidate for downstream generative tasks even if it doesn't maximize R2.
