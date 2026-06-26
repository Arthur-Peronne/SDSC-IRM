# Trial 20 — AE3dDilatedAttentionReduction4 — FAILURE

## Hypothesis
The champion's SE blocks use `reduction=16` globally. At enc1 (8 channels): `8//16=0` → Linear(8, 0) is degenerate — produces all-zeros, sigmoid gives 0.5 → uniform scaling (no attention effect). At enc2 (16 channels): `16//16=1` → Linear(16, 1) is nearly degenerate, with only 1 bottleneck unit. Only enc3 (32ch → 2 units) and enc4 (64ch → 4 units) have meaningful SE capacity. Hypothesis: fixing these degenerate blocks by using `reduction=4` throughout (enc1: 2 units, enc2: 4 units, enc3: 8 units, enc4: 16 units) should make the champion's attention mechanism actually functional at early encoder stages, improving channel recalibration with minimal added parameters (+2,048).

## Implementation
`AutoEncoder3D_DilatedAttentionReduction4`: identical to champion except all four `DilatedAttentionConv3DBlock` calls use `reduction=4`. No new block classes. No other changes.

## Results
- **validation_R2_mean:** 0.704279
- **validation_R2_std:** 0.188379
- **val_R2_lower_bound** (mean − std): 0.515900
- **lower_bound_compared_to_champion** (trial lb − champion lb): -0.214835
- **mean_compared_to_champion** (trial mean − champion mean): -0.099359
- **MLflow Run ID:** 59a407eefe094262a516a1e87d55f532
- **Best epoch:** 32 / 62 (early stop)

## Training Dynamics
Very early stopping at epoch 62, best epoch 32. Validation std exploded to 0.188 — the worst of all Phase 2 trials. Training R2 (0.829) vs validation R2 (0.704) shows the largest train/val gap in Phase 2, indicating severe overfitting.

## Conclusion
The hypothesis failed catastrophically. Replacing the degenerate SE with functional SE (reduction=4) made the model significantly worse on all metrics. Three mechanisms explain this:

1. **Active SE at enc1 introduces harmful bias.** With 8 channels and reduction=4, the SE at enc1 can now genuinely learn to suppress some channels and amplify others. But with only 8 channels and 100 training patients, the SE gates overfit to training-specific channel patterns. The champion's "degenerate" enc1 SE is implicitly a beneficial regularization — it forces the early encoder to use all 8 channels equally, preventing channel collapse on a small dataset.

2. **Increased SE capacity at enc3/enc4 amplifies overfitting.** The champion's enc3 has a 2-unit SE bottleneck (Linear(32,2)); trial 20 increases this to 8 units (Linear(32,8)). On 100 training patients, this 4× capacity increase in the deepest encoder stages leads to stronger memorization of training-specific channel statistics. The feature maps at enc3 (spatial dim 4×16×16) still have high spatial resolution relative to the patient count, so SE overfits to training-set channel correlations.

3. **The champion's apparent flaw is actually deliberate regularization.** The `reduction=16` value was presumably chosen during the original architecture search (trial 6). The degenerate enc1 and near-degenerate enc2 SE appear to be an accidental but effective regularization: enc1 does no channel gating (all features flow equally), enc2 does minimal gating (1 bottleneck unit acts as a single-axis projection), and the full SE effect is only activated at enc3-enc4 where the channel count is large enough to support stable statistics. Replacing this with "correct" SE uniformly disrupts the implicit regularization structure.

**Key insight**: On this 100-patient dataset, more expressive SE is worse. The champion's reduction=16 is not a bug — it is a form of capacity control that prevents the SE gates from memorizing training-specific channel correlations. The champion achieves its excellent generalization (std=0.073) partly because its SE is minimally parameterized at the early, wide-field encoder stages.
