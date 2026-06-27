# Existing Architecture — AE3dLinear — REFERENCE

## Architecture Description
`AE3dLinear` is a purely linear autoencoder: the encoder flattens the 3D volume and applies a single linear projection (no activation function) to the latent space; the decoder applies a single linear projection back to full resolution. There are no hidden layers, no nonlinearities, no convolutions. This makes it mathematically equivalent to truncated PCA — the optimal linear dimensionality reduction for reconstruction MSE — with the key difference that PCA solves it analytically while AE3dLinear learns it via gradient descent.

## Results
- **R2_dim8:** 0.456944 | **R2_dim60:** 0.568766 | **R2_dim240:** 0.745862
- **avg_validation_R2_mean:** 0.590524
- **delta_vs_champion:** — (REFERENCE run)
- **MLflow Run IDs:** d3cd4b195aee4d78943cd8eb4a897b38 fb34b6cd9d064492adafa7455e85c82e de3854f2406f485b9b32a4730bd4ba8f

## Training Dynamics
AE3dLinear required many more epochs than other architectures:
- dim 8: ran to near-max epochs (~300), very slow convergence
- dim 60: early stopping at epoch 205 (best: epoch 175)
- dim 240: ran to near-max epochs, still improving slowly at termination

The LR scheduler fired many times across all dims, with the final LR decaying to ~1e-7 — far below the initial 5e-5. This progressive decay is consistent with gradient descent converging to the PCA solution, which lies in a flat region of the loss surface relative to random initializations. The slow convergence arises because gradient descent on a linear model must effectively perform power iteration through backprop, which is significantly less efficient than the SVD-based PCA solver.

## Conclusion
`AE3dLinear` scores 0.5905 on average — by far the lowest in the sweep. The steep performance gap at low latent dims (dim 8: 0.457, dim 60: 0.569) versus dim 240 (0.746) confirms that cardiac MRI data has strong nonlinear structure that linear projections cannot capture at low dimensions. At dim 8, even PCA can only explain ~46% of validation variance — the 8 principal directions are insufficient to represent the full diversity of cardiac shapes and pathologies. The relatively decent dim 240 performance (0.746) shows that with enough linear directions, linear models can reconstruct most of the variance, but they are still outperformed by all nonlinear architectures tested. This architecture serves as the linear baseline: any model scoring below ~0.59 average would be doing worse than pure PCA.
