---
model_name: AE3dAsymResSeparableV2
summary: Baseline HPs as currently configured (lr=5e-5, weight_decay=0.0, dropout_rate=0.0,
  noise_std=0.0, patience=30); procedural smoke test (n_epochs=5, not scientifically
  meaningful)
parent: null
id: 21acba9e
status: completed
verdict: BASELINE
created_at: '2026-07-10T18:46:27+00:00'
metric:
  primary:
    name: avg_validation_R2_mean
    value: -0.542756
    direction: maximize
---

# Trial 21acba9e — AE3dAsymResSeparableV2 — BASELINE

<!-- ===== written BEFORE the run (agent) ===== -->

## Hypothesis
No champion exists yet in this campaign's ledger (`trial_log.csv` is empty), so this
trial is not a hyperparameter change but a reference point: run the AE3dAsymResSeparableV2
architecture with the hyperparameters already committed in `configs/autoencoder.yaml`
(lr=5e-5, weight_decay=0.0, dropout_rate=0.0, noise_std=0.0, patience=30), unmodified.
Purpose: (1) establish the BASELINE row the next two trials will be compared against,
and (2) verify end-to-end that the driver (lock -> train x3 dims -> read MLflow by tag
-> aggregate -> verdict -> commit x2) runs without a mechanical bug. `n_epochs` is 5 in
this campaign's config, so the R² values themselves are not scientifically meaningful —
only the pipeline's correctness is being checked with this trial.

## Implementation
No change to `configs/autoencoder.yaml` relative to HEAD: this trial trains the current
committed configuration as-is, over `repeat_over: {latent_dimensions: [8, 40, 200]}`.

<!-- ===== written AFTER the run ===== -->

## Results
- **validation_R2_mean per run:** 8: -0.192071 | 40: -0.274302 | 200: -1.161894
- **avg_validation_R2_mean:** -0.542756
- **delta_vs_champion** (display only): +0.000000
- **validation_MSE_mean** (mean, non-decisional): 946.563293
- **MLflow Run IDs:** 24147bded24348258f1199327d617f58 1f0d9b413b0b4db785b514190269b2d9 6c36e76be26a479aa991ddfef1e2463b

## Training Dynamics
All three runs (dims 8/40/200) show smooth, monotonic descent of both train and val
loss at every epoch, no spikes or divergence. dim=8 and dim=200 hit a new best at all
5/5 epochs; dim=40 hit a new best for 4/5 epochs then flattened on epoch 5 (still
within `patience=30`, so early stopping never triggers — as expected with only 5
epochs). Higher latent_dimensions trains to a lower absolute loss value in fewer
relative steps (dim=200 val loss drops from 0.1405 to 0.0048) but its final
validation_R2_mean is the worst of the three (-1.1619 vs -0.19 at dim=8): with only 5
epochs the higher-capacity model is comparatively more undertrained relative to its
parameter count, consistent with R² (variance-explained) being harsher on undertrained
high-dimensional reconstructions than raw MSE is.

## Conclusion
Hypothesis held on both counts. (1) Pipeline correctness: commit 1 locked the input,
all 3 trainings were tagged `trial_id=21acba9e` and read back correctly by MLflow
search, the mean aggregation and BASELINE verdict (no prior champion) were computed
and written correctly, and commit 2 froze the result — no mechanical bug anywhere in
lock -> train -> measure -> decide -> commit. (2) Scientific irrelevance: as
predicted, `avg_validation_R2_mean=-0.5428` (worse than predicting the mean everywhere)
confirms 5 epochs is far too few to fit this architecture — these numbers should not
be read as an architecture/hyperparameter verdict, only as the campaign's structural
BASELINE row. The pipeline is validated and ready for trial 2.