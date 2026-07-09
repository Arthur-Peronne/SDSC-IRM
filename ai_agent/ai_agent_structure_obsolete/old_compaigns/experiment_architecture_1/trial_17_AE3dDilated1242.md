# Trial 17 — AE3dDilated1242 — FAILURE

## Hypothesis
I will replace the champion's dilation pattern (1,2,4,1) with (1,2,4,2), changing only enc4's dilation from 1 to 2. This is a zero-parameter modification — the same DilatedAttentionConv3DBlock class is reused with a different dilation argument. The champion's enc4 returns to local features (d=1) after enc3's widest receptive field (d=4). I hypothesized that maintaining the progressive pattern with d=2 at enc4 (instead of collapsing back to local) would give the deepest encoder stage a larger receptive field before the bottleneck, helping capture broader spatial patterns before compression.

## Implementation
`AutoEncoder3D_Dilated1242`: identical to champion except `enc4 = DilatedAttentionConv3DBlock(32, 64, dilation=2, ...)` instead of `dilation=1`. No new classes or parameters introduced.

## Results
- **validation_R2_mean:** 0.756790
- **validation_R2_std:** 0.131438
- **val_R2_lower_bound** (mean − std): 0.625352
- **lower_bound_compared_to_champion** (trial lb − champion lb): -0.105383
- **mean_compared_to_champion** (trial mean − champion mean): -0.046848
- **MLflow Run ID:** edec4b92edf746b38256935669408619
- **Best epoch:** 45 / 75 (early stop)

## Training Dynamics
Early stopping at epoch 75 (best epoch 45). Validation std increased sharply to 0.131 (vs champion 0.073), indicating the dilation change introduced more inter-patient variance. Training R2 (0.884) vs validation R2 (0.757) shows a larger train/val gap than the champion.

## Conclusion
The hypothesis failed. Changing enc4's dilation from 1 to 2 degraded both mean and variance. The champion's return to d=1 at enc4 is a deliberate structural choice:

1. **Local feature consolidation before bottleneck.** After three stages of progressively wider dilation (1→2→4), enc4 with d=1 consolidates these multi-scale representations into local feature descriptors. The bottleneck then compresses these consolidated local features. Maintaining d=2 at enc4 skips this consolidation step, sending multi-scale (not locally consolidated) features directly into the bottleneck, which cannot efficiently compress them.

2. **Feature map size at enc4.** After three MaxPools on 128×128→64×64→32×32→16×16 spatial dims, enc4 processes 4×16×16 features. With d=2, the effective kernel span is 5 pixels (2×(k-1)+1=2×2+1=5 out of 16). The champion's d=1 gives a 3-pixel effective span, which is appropriate for local consolidation at this scale. The d=2 span captures too wide a neighborhood, introducing redundant information at this already-compressed resolution.
