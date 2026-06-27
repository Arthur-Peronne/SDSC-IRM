# Existing Architecture — AE3dCurrent — REFERENCE

## Architecture Description
`AE3dCurrent` is the standard convolutional 3D autoencoder. The encoder applies successive 3D convolutions with max-pooling to progressively reduce spatial resolution, then flattens the feature maps into a 1D vector that is projected to the latent space via a fully-connected layer. The decoder mirrors this with a linear projection followed by transposed convolutions and trilinear upsampling. No skip connections, no attention, no dilation — a direct bottleneck architecture.

## Results
- **R2_dim8:** 0.713262 | **R2_dim60:** 0.732954 | **R2_dim240:** 0.693351
- **avg_validation_R2_mean:** 0.713189
- **delta_vs_champion:** — (first REFERENCE run)
- **MLflow Run IDs:** efc10e1740eb4bed8ec77e88d5bb2757 2a409469cc5f4c84861db97f1405eeb6 a4e9410aeeda47f4b66f967fee036d5f

## Training Dynamics
All three dims converged cleanly with early stopping:
- dim 8: best at epoch ~37, stopped at epoch ~67
- dim 60: best at epoch ~50, stopped at epoch ~80
- dim 240: best at epoch ~40, stopped at epoch ~70

Learning rate scheduler fired as expected (halving on plateau), with no instability or loss spikes.

## Conclusion
`AE3dCurrent` achieves a solid R2 baseline across all three latent dims. Notably, R2 peaks at dim 60 (0.733) and drops at dim 240 (0.693), suggesting that the standard conv architecture over-smooths reconstructions at very high latent capacity — likely due to its limited receptive field failing to capture fine-grained spatial detail that would benefit from larger latent codes. The dim 8 performance (0.713) is competitive, indicating the architecture compresses cardiac structure efficiently at low dimensionality.
