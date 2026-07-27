---
model_name: AE3dAsymResSeparableV2
summary: Increase learning rate from 5e-4 to 8e-4 on champion architecture, testing HP tuning direction after 7 failed architectural modifications
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
Eight consecutive trials (b606a10f, 0cadad28, 001b5a08, e4e99f58, 25eae42a, 54805b0f, 54805b0f, b27081bf) have failed by manipulating architecture (channel width, weight_decay, dilated attention, standard+attention, bottleneck attention). The consistent failures suggest that architectural modifications are not the lever that improves classification_accuracy_val.

This trial pivots to **hyperparameter tuning on the champion architecture**. The previous HP campaign (aiagent_HP_sepv2_240) on the sepv2 architecture found that lr=8e-4 + dropout=0.05 was the optimal combination, significantly outperforming lr=5e-4 (baseline). While that campaign was on a different architecture, the learning rate is a fundamental training parameter that affects optimization dynamics regardless of architecture.

The champion uses lr=5e-4, which was the initial default. The HP campaign found that lr=8e-4 converges faster and reaches a better optimum. The mechanism: a higher learning rate allows the optimizer to escape shallow local minima and find a better region of the loss landscape. Combined with the champion's dropout=0.3 (which provides regularization), lr=8e-4 should produce a latent representation that is both well-trained and well-generalized.

I predict classification_accuracy_val will exceed the champion's 0.6083.

## Implementation
In `configs/autoencoder.yaml` only:
- `lr: 5e-4 -> 8e-4` (60% increase)
- All other HPs identical to champion: weight_decay=1e-5, dropout_rate=0.3, noise_std=0.0, patience=20
- model_name unchanged: AE3dAsymResSeparableV2
- No architecture file touched this trial — isolates the lr effect alone.