# Trial 25 — AE3dDilatedAttentionLatentNorm — FAILURE

## Hypothesis
All 25 trials (including those from previous sessions) have failed to reduce inter-patient reconstruction variance below the champion's std=0.073. Trials 21–22 showed that the bottleneck/decoder (~1.8M params) dominates variance, not the encoder (~220K). Trial 25 hypothesizes that inter-patient variance in val R2 is partly driven by variability in latent vector statistics across patients: some patients produce larger-magnitude latent vectors (easier to reconstruct) while others produce smaller ones (harder). LayerNorm applied to z after fc_enc (normalizing the 120-dim latent to zero mean and unit variance per patient) should force consistent latent statistics across all patients, potentially reducing reconstruction variance. +240 params (LayerNorm weight + bias for 120 dims).

## Implementation
`AutoEncoder3D_DilatedAttentionLatentNorm`: champion encoder (DilatedAttentionConv3DBlock enc1–enc4, dilations 1/2/4/1, reduction=16) and bottleneck unchanged. After fc_enc: `self.latent_norm = nn.LayerNorm(latent_dim)` applied to z. The decode path receives the normalized z directly — no additional normalization. Total params: 2,022,097 (+240 vs champion). Validated: z mean=0.0, z std≈1.0 for random input (BatchNorm normalizing 120 dims per sample).

## Results
- **validation_R2_mean:** 0.754723
- **validation_R2_std:** 0.124091
- **val_R2_lower_bound** (mean − std): 0.630632
- **lower_bound_compared_to_champion** (trial lb − champion lb): -0.100103
- **mean_compared_to_champion** (trial mean − champion mean): -0.048915
- **MLflow Run ID:** 79817ee471ab4f89b8a0b4eaabf62f21
- **Best epoch:** 43 / 73 (early stop)

## Training Dynamics
Early stopping at epoch 73 (best epoch 43). Train R2=0.854 vs val R2=0.755 — clear train/val gap. std=0.124, 1.7× the champion's 0.073. The LayerNorm on the latent did not help.

## Conclusion
The hypothesis failed. Per-patient latent normalization does not reduce inter-patient reconstruction variance.

Two explanations for the failure:

1. **LayerNorm destroys relative latent magnitude information.** The absolute magnitude of z (before normalization) may encode patient-level information such as cardiac size, image contrast, or frame quality. LayerNorm zeroes out this per-patient scale, forcing the decoder to reconstruct without access to magnitude information it would normally use. This forces the decoder to work harder from directional information alone.

2. **The variance is in the decoder's spatial computation, not the latent statistics.** The champion's low std comes from how the decoder transforms the latent back to 3D MRI — specifically the convolution weights that determine spatial reconstruction quality. Normalizing the INPUT to the decoder (z) does not change how the decoder handles spatially varying patient anatomy. The variance source is in the decoder's convolutional layers, not in the latent scale.

**Additional observation:** The best epoch (43) and stopping epoch (73) are similar to the FAILURE pattern across most trials — the network converges quickly then plateaus without finding the narrow optimum the champion occupies. The champion likely has a specific feature geometry (dilated encoder + degenerate SE at enc1/enc2 + specific bottleneck depth) that creates favorable gradient dynamics. Adding LayerNorm disrupts this by clipping information during the latent computation.

**Lesson:** Latent space interventions (LayerNorm in trial 25, 1×1×1 bottleneck compression in trial 16) consistently fail. The effective latent dimension and the way information flows through fc_enc → latent → fc_dec appear to be optimized by the champion's existing architecture. The only beneficial modification found so far remains trial 15's SE channel calibration in the decoder.
