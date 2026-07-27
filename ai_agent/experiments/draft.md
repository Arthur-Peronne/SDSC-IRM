---
model_name: AE3dAsymResSeparableV2
summary: lr=8e-4 + dropout=0.1 (reduced dropout to match higher lr, per trial 11's coupling insight)
parent: 3aa0388f
id: null
status: draft
verdict: null
created_at: null
metric:
  primary: {name: avg_validation_R2_mean, value: null, direction: maximize}
---

# Trial <id> — <model_name> — <verdict>

## Hypothesis
Trial 11 (4682d5a1) found lr=8e-4 alone gave classification_accuracy_val=0.5917, very close to champion's 0.6083 but within noise. The key insight: validation_R2_mean=0.735 (best of campaign) confirms higher lr improves reconstruction, but the classification improvement is marginal — likely because the champion's dropout=0.3 is too aggressive when combined with higher lr.

This trial tests **lr=8e-4 + dropout=0.1**: reducing dropout from 0.3 to 0.1 while keeping the higher learning rate. The mechanism: higher lr allows faster convergence to a better optimum, while lower dropout preserves more information in the bottleneck representation (less regularization noise). The two HPs are coupled — the previous HP campaign found lr=8e-4 + dropout=0.05 was optimal, and lr=5e-4 + dropout=0.3 was suboptimal.

I predict classification_accuracy_val will exceed the champion's 0.6083.

## Implementation
In `configs/autoencoder.yaml` only:
- `lr: 5e-4 -> 8e-4` (60% increase)
- `dropout_rate: 0.3 -> 0.1` (67% decrease)
- All other HPs identical to champion: weight_decay=1e-5, noise_std=0.0, patience=20
- model_name unchanged: AE3dAsymResSeparableV2
- No architecture file touched this trial.