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
Early stopping at epoch 70/40/69 (seeds 0/1/2), best epoch 50/20/49 (val loss 0.004186/0.003340/
0.004162) — noticeably longer and more variable than the champion's 29/38/32: two of three seeds
took ~20 epochs longer to plateau. No NaNs or divergence, but the optimization landscape is visibly
harder/slower with the added nonlinear FC depth.

## Conclusion
Hypothesis did not hold. Both metrics dropped together this time: `classification_accuracy_val`
0.700000 -> 0.641667 (-0.058, FAILURE, per-seed 0.675/0.65/0.60 — no seed reached the champion's
range), and `validation_R2_mean` 0.7321 -> 0.7016 (also down, unlike the pooling trial where R2 was
unaffected). Both drops exceed the noise floor.

Mechanistic read: this trial added real capacity (~1.13M -> ~1.58M params) with zero regularization
(`weight_decay=0`, `dropout_rate=0`, unchanged across all architecture trials to isolate the
architecture variable) on a small dataset (100 training patients, batch_size=1). The slower,
more variable convergence supports an optimization-difficulty / mild-overfitting story rather than a
capacity-is-fundamentally-wrong one: extra unregularized parameters have more room to fit
idiosyncrasies of the training set that don't generalize, for both reconstruction and classification.

Pattern across the last 3 architecture trials worth flagging explicitly: the two that *added*
complexity/capacity without regularization (CBAM spatial attention, this nonlinear FC) both failed
on classification accuracy (one with R2 preserved, one with R2 also down); the one that *removed*
redundant capacity (SE late-only ablation) is the campaign's best result on both metrics. This
suggests the champion may currently be regularization-starved rather than capacity-starved — a
legitimate hypothesis to test directly next via a small, targeted regularization check on the
existing champion architecture (no new model code), before proposing further architecture growth.
Champion remains 761cab78. Code reverted by the driver.