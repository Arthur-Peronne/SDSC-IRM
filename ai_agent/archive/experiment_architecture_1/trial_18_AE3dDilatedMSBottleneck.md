# Trial 18 — AE3dDilatedMSBottleneck — FAILURE

## Hypothesis
I will replace each of the champion's two bottleneck conv layers with a `MultiScaleBottleneckLayer`: two parallel Conv3d branches (d=1 and d=2), each outputting 64 channels, concatenated to 128. This keeps the parameter count identical to the champion while the bottleneck now encodes features at two different receptive field sizes simultaneously. The hypothesis: a multi-scale bottleneck captures both local (d=1) and slightly global (d=2) structure before the FC projection, producing richer latent representations.

## Implementation
New `MultiScaleBottleneckLayer(in_ch, out_ch)`: `branch1 = Conv3d(in_ch, out_ch//2, k=3, d=1, p=1)` and `branch2 = Conv3d(in_ch, out_ch//2, k=3, d=2, p=2)`, concatenated along channel dim, followed by `InstanceNorm3d(out_ch) + ReLU`. Two such layers replace the champion's sequential `bottleneck_conv`. Encoder and decoder are unchanged.

## Results
- **validation_R2_mean:** 0.764956
- **validation_R2_std:** 0.155076
- **val_R2_lower_bound** (mean − std): 0.609881
- **lower_bound_compared_to_champion** (trial lb − champion lb): -0.120854
- **mean_compared_to_champion** (trial mean − champion mean): -0.038682
- **MLflow Run ID:** 97bc46d33b9c45b2b553d1f6b5b9c9a3
- **Best epoch:** 48 / 78 (early stop)

## Training Dynamics
Early stopping at epoch 78 (best epoch 48). Validation std exploded to 0.155 — more than double the champion's 0.073 — producing the joint-worst lower bound in the exploration phase (after GroupNorm's 0.548). Training R2 (0.868) vs validation R2 (0.765) shows a clear train/val gap.

## Conclusion
The hypothesis failed. Despite identical parameter count, the multi-scale bottleneck degraded performance significantly. Two mechanisms explain this:

1. **Branch interference during concatenation.** The d=1 and d=2 branches produce features at different effective receptive fields. When concatenated and passed through InstanceNorm as a combined 128-channel tensor, the normalization statistics mix features from two very different spatial scales. This creates incoherent feature distributions in the subsequent FC projection — the two halves of the 2048-dim flattened vector have different statistical properties, making the linear mapping to the 120-dim latent space harder to learn.

2. **The bottleneck spatial resolution is too small for meaningful multi-scale.** After `final_down`, the feature map is (128, 1, 4, 4). Applying dilation=2 on a 1×4×4 spatial map means the d=2 branch looks at positions {0, 2, 4} in H and W — which spans nearly the entire feature map. The "global" branch (d=2) and "local" branch (d=1) have very similar effective receptive fields at this resolution, providing little additional information while complicating learning.

The champion's plain d=1 bottleneck is optimal for this spatial scale: it performs local feature mixing at (2, 8, 8) before the final_down step.
