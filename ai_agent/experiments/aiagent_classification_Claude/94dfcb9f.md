---
model_name: AE3dAsymResSeparableV2SELateEnc3Only
summary: 'HP tuning on champion architecture (no code change): patience 20 -> 35,
  testing whether the recurring weak-seed pattern improves with more room to escape
  an early plateau'
parent: ac5057cf
id: 94dfcb9f
status: completed
verdict: FAILURE
created_at: '2026-07-29T06:02:10+00:00'
metric:
  primary:
    name: classification_accuracy_val
    value: 0.691667
    direction: maximize
---

# Trial 94dfcb9f — AE3dAsymResSeparableV2SELateEnc3Only — FAILURE

<!-- ===== written BEFORE the run (agent) ===== -->

## Hypothesis
Only 4 trials remain (`lr` is closed both directions, two fusion attempts and one SE-refinement have
all failed on top of champion `ac5057cf`) — `patience` (currently 20) is the one hyperparameter never
touched this entire campaign. A recurring pattern across recent trials (`ac5057cf`, `b02b8293`,
`0afac601`, `582021d1`) is "two seeds land near 0.70-0.75, one seed lands notably weaker," and several
weak seeds show fast, shallow convergence (e.g. `582021d1`'s seed 0, best epoch 13) — consistent with
early stopping cutting off a seed before it has a chance to escape an early plateau via the LR-decay
schedule (which only kicks in after `patience` epochs of no improvement, and decays further every
`patience//5` epochs after that). Raising `patience` to 35 gives every seed more LR-decay steps before
stopping, which I predict either helps the weak seeds catch up (improving the aggregate) or is
neutral if `ac5057cf`'s champion result already reflects each seed's true achievable optimum within
its own dynamics. No code change — pure hyperparameter test.

## Implementation
No new class — reuses `AutoEncoder3D_AsymResSeparableV2_SELateEnc3Only` (model_name
`AE3dAsymResSeparableV2SELateEnc3Only`), the exact champion `ac5057cf` code. Only
`configs/autoencoder.yaml` changes: `patience: 20 -> 35`. All other hyperparameters (`lr=1e-4`,
`weight_decay=0`, `dropout_rate=0`, `noise_std=0`) and the architecture itself are byte-for-byte the
champion's.

<!-- ===== written AFTER the run ===== -->

## Results
- **accuracy_test per run:** 0: 0.725000 | 1: 0.675000 | 2: 0.675000
- **classification_accuracy_val:** 0.691667
- **delta_vs_champion** (display only): -0.025000
- **validation_R2_mean** (mean, AE phase, non-decisional): 0.725194
- **AE MLflow Run IDs:** bbc05675e07b47f19a9f69bdec9083a0 a84c6b1dc06c405da1043e19042f5f74 bd1a6dc595a84b0fb06827863831bb23
- **Classification MLflow Run IDs:** 9f83af54c3e242d8977928b07f4ae85d 1f548f60d0ec48649b65c6a2b8ad197c f8c0285af13945f4a703261a0397868e

## Training Dynamics
Early stopping at epoch 55/78/73 (seeds 0/1/2), best epoch 20/43/38 — with `patience=35`, all three
seeds ran meaningfully longer than the champion's 32/32/38, giving each more LR-decay steps as
predicted. No NaNs. But the extra training time shifted each seed to a *different* optimum, not
uniformly a *better* one: seed 0 actually stopped earlier in relative terms (best epoch 20, its
weakest R² of 0.6627) despite the longer patience budget, while seeds 1/2 used the extra room
(best epochs 43/38) and reached slightly better R² than under `patience=20`. `validation_R2_mean`
(0.7252) came in below the champion's 0.7432 despite more training per seed.

## Conclusion
Hypothesis not supported. More patience did not systematically help the weaker seed catch up — seed 0
remained the relative underperformer (per-seed accuracy 0.725/0.675/0.675, delta -0.025) despite
having just as much room to keep improving as the other two. This suggests the "two strong, one weak"
pattern noted across recent trials is not simply an artifact of premature early stopping — extending
the training budget shifts *where* each seed's optimizer lands, but doesn't reliably rescue the weaker
seed, reinforcing this campaign's working read that this variance is closer to an inherent property of
this task's 3-seed sample at `n_train=100` than a fixable training-schedule issue. `patience` is now
closed alongside `lr`: neither hyperparameter lever tested this campaign has beaten the champion.
`ac5057cf` remains the campaign champion (classification_accuracy_val=0.7167,
validation_R2_mean=0.7432) as this campaign is paused per the user's request after this trial.