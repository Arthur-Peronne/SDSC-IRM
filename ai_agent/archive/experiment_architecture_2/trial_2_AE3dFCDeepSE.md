# Trial 2 — AE3dFCDeepSE — FAILURE

## Hypothesis
Replace AE3dFCDeep's four standard Conv3DBlocks (enc1–enc4) with AttentionConv3DBlocks — Squeeze-and-Excitation (SE) channel attention with reduction=4 applied after each MaxPool downsampling step. Bottleneck convolutions, final_down, FC layers, and decoder remain identical to AE3dFCDeep. Rationale: Trial 1 showed that dilated convolutions hurt at dim=8 by expanding spatial receptive fields before MaxPool, degrading feature compactness. SE attention operates differently — it recalibrates *channel weights* after MaxPool, leaving spatial resolution and receptive field unchanged. The hypothesis was that SE would help the FC bottleneck at all dims by upweighting the most informative feature channels before the linear compression step, without the spatial disruption introduced by dilation.

## Implementation
- enc1: AttentionConv3DBlock(1→8, downsample=True, reduction=4) — SE over 8 channels with 2-neuron squeeze
- enc2: AttentionConv3DBlock(8→16, downsample=True, reduction=4) — SE over 16 channels with 4-neuron squeeze
- enc3: AttentionConv3DBlock(16→32, downsample=True, reduction=4) — SE over 32 channels with 8-neuron squeeze
- enc4: AttentionConv3DBlock(32→64, downsample=True, reduction=4) — SE over 64 channels with 16-neuron squeeze
- All other layers (bottleneck_conv, final_down, FC, decoder) identical to AE3dFCDeep
- Total params: 1,778,085 (slightly larger than AE3dFCDeep due to SE linear layers)

## Results
- **R2_dim8:** 0.748674 | **R2_dim60:** 0.726034 | **R2_dim240:** 0.723713
- **avg_validation_R2_mean:** 0.732807
- **delta_vs_champion** (trial avg − champion avg): −0.018902
- **MLflow Run IDs:** 04f1ec6b74e148eabd009543b8f8a27b 010562da017940cbae60676408162a67 e76f2d6a171944db86ab7dac636a94a1
- **Best epochs:** 47/300 | 47/300 | 48/300

## Training Dynamics
All three runs converged stably with no loss spikes. Early stopping triggered consistently early: epoch 77 (dim=8), epoch 77 (dim=60), epoch 78 (dim=240) — all noticeably earlier than the champion's convergence (which typically required 100–150 epochs before plateau). The learning rate scheduler halved repeatedly and settled at ~1.5e-6 by termination, suggesting the model reached a genuine local minimum and did not benefit from extended training. This early convergence pattern hints that SE attention may reduce the encoder's effective capacity, causing the model to saturate earlier.

## Conclusion
The hypothesis did not hold. SE channel attention uniformly degraded performance across all three latent dimensions:
- **dim=8:** 0.7487 vs champion 0.7715 (Δ=−0.023) — moderate regression
- **dim=60:** 0.7260 vs champion 0.7258 (Δ=+0.000) — statistically indistinguishable
- **dim=240:** 0.7237 vs champion 0.7578 (Δ=−0.034) — largest regression

The dim=240 regression is the most revealing result. Unlike Trial 1 where dilation hurt specifically at small dims and was neutral at large dims, SE attention hurts most at large dims. This reveals the mechanism: SE channel attention is a *soft information bottleneck* — it suppresses channels judged globally unimportant by the squeeze (global average pooling) operation. In a reconstruction autoencoder at high latent dims (240), the decoder can use many independent codes to reconstruct fine detail. Suppressing even moderately informative encoder channels removes signal that would otherwise contribute to this high-dimensional reconstruction. The loss is disproportionate because at large dims, the decoder expects a rich latent space — any channel gating reduces the effective rank of the feature map entering the FC bottleneck.

At dim=8, the FC layer is already the dominant bottleneck (compressing 2048→8), so SE's contribution is relatively smaller. At dim=60, the two effects roughly cancel. At dim=240, SE's channel suppression becomes the binding constraint.

A secondary issue: enc1's SE squeeze uses only 2 neurons (8 channels / reduction=4). Such a narrow squeeze layer cannot represent meaningful channel-wise correlations and may introduce noise. Future architectures using SE should either use higher channel counts in the early layers or use a larger reduction ratio to avoid degenerate squeezes.

**Verdict:** SE channel attention in reconstruction autoencoders selectively suppresses encoder channels that may carry globally weak but locally critical spatial information. Unlike classification networks where SE consistently improves performance by focusing on task-discriminative channels, reconstruction networks require broad feature coverage — SE creates an implicit capacity ceiling that hurts most when the latent space is large enough to exploit diverse encoder features.
