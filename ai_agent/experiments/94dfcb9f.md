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
<!-- Agent, after the run: stability, convergence speed, spikes, plateau, early stopping. -->

## Conclusion
<!-- Agent, after the run: did the hypothesis hold? Mechanistic explanation of why it
     worked or failed — not just the numbers. -->