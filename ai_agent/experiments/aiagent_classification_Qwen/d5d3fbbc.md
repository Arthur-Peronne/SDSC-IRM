---
model_name: AE3dAsymResSeparableV2
summary: Increase patience 20->40 on champion (lr=8e-4, dropout=0.3, weight_decay=1e-5),
  testing if higher early stopping patience allows fuller convergence
parent: a2a3d9d1
id: d5d3fbbc
status: completed
verdict: FAILURE
created_at: '2026-07-28T02:18:49+00:00'
metric:
  primary:
    name: classification_accuracy_val
    value: 0.525
    direction: maximize
---

# Trial d5d3fbbc — AE3dAsymResSeparableV2 — FAILURE

<!-- ===== written BEFORE the run (agent) ===== -->

## Hypothesis
Patience is the ONLY hyperparameter never tested from its default value (20) in this campaign. The champion (`lr=8e-4`, `dropout=0.3`) converges relatively quickly (early stopping at epochs ~50-67). With `patience=20`, the network stops 20 epochs after the best validation metric. If the validation metric has natural noise (program.md notes ~0.03-0.4 std dev on R²), the early stopping might trigger too early, before the network has fully converged. Doubling patience to 40 would give the network more epochs to find a better minimum, while the early stopping mechanism still prevents overfitting. I predict classification_accuracy_val ≥ 0.6250, with potentially higher R² due to fuller convergence.

## Implementation
In `configs/autoencoder.yaml` (only mutable file), branching from champion a2a3d9d1:
- `patience: 20 -> 40` (early stopping patience doubled)
- Unchanged: `lr=8e-4`, `dropout_rate=0.3`, `weight_decay=1e-5`, `noise_std=0.0`, `model_name=AE3dAsymResSeparableV2`, `latent_dimensions=20`

<!-- ===== written AFTER the run ===== -->

## Results
- **accuracy_test per run:** 0: 0.550000 | 1: 0.450000 | 2: 0.575000
- **classification_accuracy_val:** 0.525000
- **delta_vs_champion** (display only): -0.100000
- **validation_R2_mean** (mean, AE phase, non-decisional): 0.740447
- **AE MLflow Run IDs:** 74d1df7e795a4f3ea692ecc27fbce551 996891b77c0f4d14b1a4ed9336854824 7550ede295f742238c8dfb88881328e7
- **Classification MLflow Run IDs:** bb942484140e465d914cf8b5e2e9842c 348301526d4b459d987b63783d56efba 7377a9b05afa48279c3062565f766575

## Training Dynamics
Moderate failure across all 3 seeds. R² = 0.7404 is close to champion's 0.7567, suggesting the network still learns reasonable reconstructions with more epochs. However, the accuracy drop (0.5250 vs 0.6250) shows that more training epochs don't help the classifier.

## Conclusion
Hypothesis **rejected**. `patience=40` + `lr=8e-4` + `dropout=0.3` gives 0.5250, worse than the champion's 0.6250. The higher patience doesn't help — the network with `patience=20` was already converging to the same or better solution. This suggests the network reaches its optimal point within the original patience window, and extra epochs don't provide additional benefit. **Important**: patience is not a lever worth exploring. The champion's `patience=20` is already well-calibrated. From now on, I should focus on architectural changes, as HPs are now well-explored (lr=8e-4, dropout=0.3, weight_decay=1e-5, patience=20 are all near-optimal).