# Trial 1 — AE3dFCDeepDilated — FAILURE

## Hypothesis
Replace AE3dFCDeep's four standard Conv3DBlocks with DilatedConv3DBlocks using a 1-2-2-1 dilation pattern (enc1=1, enc2=2, enc3=2, enc4=1). Bottleneck convolutions, final_down, FC layers, and decoder remain identical to AE3dFCDeep. Rationale: AE3dFCDeep dominates the reference sweep because its progressive compression + FC bottleneck generalises well across latent dimensions. Its weakness is a limited receptive field (~5×5×5 effective per block). AE3dDilated improved on AE3dCurrent by +0.035 R2 by expanding receptive field; combining dilated convolutions with FCDeep's superior bottleneck should yield additive benefits.

## Implementation
- enc1: DilatedConv3DBlock(1→8, dilation=1, downsample=True) — same effective receptive field as Conv3DBlock
- enc2: DilatedConv3DBlock(8→16, dilation=2, downsample=True) — expanded receptive field at 16×64×64
- enc3: DilatedConv3DBlock(16→32, dilation=2, downsample=True) — expanded receptive field at 8×32×32
- enc4: DilatedConv3DBlock(32→64, dilation=1, downsample=True) — returned to dilation=1 at 4×16×16 to avoid spatial artifacts before bottleneck
- All other layers (bottleneck_conv, final_down, FC, decoder) identical to AE3dFCDeep
- Total params: 1,775,365 (same order of magnitude as AE3dFCDeep)

## Results
- **R2_dim8:** 0.728169 | **R2_dim60:** 0.754912 | **R2_dim240:** 0.765790
- **avg_validation_R2_mean:** 0.749624
- **delta_vs_champion** (trial avg − champion avg): −0.002085
- **MLflow Run IDs:** aa283949ef8e4eea8d549dde5314382b 69c79117cac148b2a529a366a60d3bb8 544a4f622a9e481ea242c5a7ca70d83f
- **Best epochs:** 52/300 | 62/300 | 44/300

## Training Dynamics
All three runs converged stably with no spikes or loss explosions. The learning rate scheduler reduced smoothly across all runs (5e-5 → 1.56e-6 by early stopping). Early stopping triggered at epochs 82, 92, and 74 for dims 8, 60, 240 respectively — consistent with AE3dFCDeep's convergence behaviour. No instability introduced by dilation.

## Conclusion
The hypothesis did not hold. The dilated encoder hurt performance specifically at dim=8 (0.728 vs champion 0.772, Δ=−0.044) while providing modest gains at dim=60 (+0.029) and dim=240 (+0.008). The asymmetric dimension-dependent impact reveals a structural mismatch: at very small latent dimensions (dim=8), the bottleneck is the binding constraint — the FC layer must compress 2048 features to 8 scalars. In this regime, what matters most is the discriminative quality of the feature map entering the FC layer. Standard Conv3DBlocks with MaxPool produce compact, translation-invariant features; DilatedConv3DBlocks maintain larger spatial coverage but their features may be less tightly compressed before MaxPool, reducing the signal-to-noise ratio at extreme compression ratios. At larger dims (60, 240), the FC bottleneck is less constrained, and the broader receptive field of dilation benefits reconstruction. The net effect is a wash overall, with dim=8 drag pulling the average below the champion.
