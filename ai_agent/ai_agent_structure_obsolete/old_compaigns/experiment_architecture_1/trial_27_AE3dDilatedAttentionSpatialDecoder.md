# Trial 27 — AE3dDilatedAttentionSpatialDecoder — FAILURE

## Hypothesis
Trials 15, 19, 26 established that channel attention (SE) in the decoder at reduction=4 is the best mechanism found (Δlb=-0.005). Spatial attention in the decoder has not been tried. Hypothesis: the decoder could benefit from knowing WHERE in the 3D volume to focus reconstruction effort. A content-based spatial attention mask — Conv3d(ch, 1, k=1) + Sigmoid applied after each decoder block, broadcast-multiplied to re-weight voxel positions — allows the model to suppress background and focus on the cardiac region. +124 params.

## Implementation
`AutoEncoder3D_DilatedAttentionSpatialDecoder`: champion encoder + bottleneck unchanged. After each decoder block: `sa_decN = nn.Sequential(nn.Conv3d(ch, 1, kernel_size=1), nn.Sigmoid())`. Forward: `x = x * self.sa_decN(x)`. Total params: 2,021,981 (+124 vs champion).

## Results
- **validation_R2_mean:** 0.704457
- **validation_R2_std:** 0.263652
- **val_R2_lower_bound** (mean − std): 0.440805
- **lower_bound_compared_to_champion** (trial lb − champion lb): -0.289930
- **mean_compared_to_champion** (trial mean − champion mean): -0.099181
- **MLflow Run ID:** 89c5536e39c245c1a8952cd94b55808f
- **Best epoch:** 35 / 65 (early stop)

## Training Dynamics
Early stopping at epoch 65 (best epoch 35). Std=0.2637 — the worst of any trial, 3.6× the champion's 0.073. Mean R2=0.704 — also poor. The spatial attention catastrophically increased variance.

## Conclusion
The hypothesis failed catastrophically. Spatial attention in the decoder causes much higher variance than any other modification.

The root cause is multiplicative feature suppression: the sigmoid spatial mask outputs values in (0,1), so it can reduce feature values at spatial positions to near-zero. For some patients, the mask learns to suppress important cardiac regions (e.g., when a patient's cardiac position or size deviates from the training distribution), effectively zeroing out those regions in the feature map. This is a fundamental problem: the mask is learned from training data but must generalize to validation patients with different cardiac geometries.

Key contrast with SE (channel attention):
- SE masks channels globally (all spatial positions): if SE suppresses a channel, it's suppressed everywhere, but other channels can still carry the information for that region
- Spatial attention masks positions locally: if a position is suppressed, that spatial location is lost across ALL channels simultaneously — no redundancy remains

This explains why spatial attention is catastrophically worse than channel attention for reconstruction: it can create spatially dead zones where reconstruction quality collapses for atypical patients.

**Confirmed:** Spatial attention in the decoder is counter-productive. Channel attention (SE, trial 15) is the only beneficial attention type for the decoder — and only at reduction=4.

**Cooldown triggered:** 3 consecutive FAILURE trials starting with "AE3dDilatedAttention" (trials 25, 26, 27). Next trial must be Exploration (different architecture family).
