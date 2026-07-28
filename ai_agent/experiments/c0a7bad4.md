---
model_name: AE3dAsymResSeparableV2SELateDeepFC
summary: 'Nonlinear FC bottleneck: fc_enc/fc_dec through a 128-d ReLU hidden layer
  instead of one Linear'
parent: 761cab78
id: c0a7bad4
status: completed
verdict: FAILURE
created_at: '2026-07-28T16:24:15+00:00'
metric:
  primary:
    name: classification_accuracy_val
    value: 0.641667
    direction: maximize
---

# Trial c0a7bad4 — AE3dAsymResSeparableV2SELateDeepFC — FAILURE

<!-- ===== written BEFORE the run (agent) ===== -->

## Hypothesis
Third distinct architectural family this campaign (after SE placement and pooling mechanism, both
now closed directions). Champion `AE3dAsymResSeparableV2SELate`'s `fc_enc`/`fc_dec` are each a
single `nn.Linear` (2048<->20): the 20-d latent is forced to be a purely linear function of the
flattened bottleneck features, and reconstruction a purely linear expansion back. I insert a 128-d
ReLU hidden layer on both sides (2048->128->20 and 20->128->2048). Mechanism: a nonlinear
compression can express feature interactions a linear map cannot (e.g. "high in channel A AND low
in channel B" style combinations) — if any of the group-discriminative structure in the 2048-d
bottleneck features is not linearly separable, only a nonlinear projection can preserve it into the
fixed 20-d budget. This is a genuinely different lever from SE (which reshapes what's IN the
2048-d vector) or pooling (which reshapes what survives spatially) — it changes how that vector is
compressed to 20 dims. Same encoder/decoder conv path, SE placement (se3, se4), and hyperparameters
(`lr=1e-4`, `weight_decay=0`, `dropout_rate=0`, `noise_std=0`, `patience=20`) as the champion.

## Implementation
New class `AutoEncoder3D_AsymResSeparableV2_SELateDeepFC` in `src/models/ae_models.py` (model_name
`AE3dAsymResSeparableV2SELateDeepFC`), copied from `AutoEncoder3D_AsymResSeparableV2_SELate`
(current champion) with `fc_enc`/`fc_dec` changed from a single `nn.Linear` each to
`nn.Sequential(Linear(in, 128), ReLU, Linear(128, out))`. Dropout is still applied at the same points
as the champion (after flatten before `fc_enc`, after `fc_dec` before reshaping) — inert either way
since `dropout_rate=0` this campaign. Everything else (conv encoder/decoder, SE) unchanged. No
encoder-to-decoder path — respects the no-skip-connections rule in `program.md`. Verified the new
class builds, forward-passes, and produces a (1,20) latent before launching the trial (parameter
count grows from ~1.13M to ~1.58M due to the two hidden FC layers, still a small model for this
task).

<!-- ===== written AFTER the run ===== -->

## Results
- **accuracy_test per run:** 0: 0.675000 | 1: 0.650000 | 2: 0.600000
- **classification_accuracy_val:** 0.641667
- **delta_vs_champion** (display only): -0.058333
- **validation_R2_mean** (mean, AE phase, non-decisional): 0.701616
- **AE MLflow Run IDs:** 45837f70ee4c4b6881f0edeaa200abe7 ee3b79ffc77c489b9a7788b4ea53c482 88382e00f51f475bb906b828b115ff80
- **Classification MLflow Run IDs:** b49de773723a421182e30e760673a26a e125fc13f9714b109fec230a8afd28f8 b63cd2f88fd64ef392de7fff11dac5ad

## Training Dynamics
<!-- Agent, after the run: stability, convergence speed, spikes, plateau, early stopping. -->

## Conclusion
<!-- Agent, after the run: did the hypothesis hold? Mechanistic explanation of why it
     worked or failed — not just the numbers. -->