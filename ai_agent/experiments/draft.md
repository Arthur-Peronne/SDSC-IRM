---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2
summary: Campaign baseline at latent_dim=240, full budget (n_epochs=200) — HP values carried from the pipeline-smoke-test champion (lr=5e-4, weight_decay=0, dropout_rate=0, noise_std=0, patience=30)
parent: null          # lineage: `id` of the trial whose CODE this one branched FROM. null for roots.

# ---- driver-written (leave null; the driver overwrites at lock/result) ----
id: null              # short sha of commit 1 == the frozen input == this trial's identity
status: draft         # draft -> completed | failed          (lifecycle, lowercase)
verdict: null         # BASELINE | CHAMPION | CANDIDATE | FAILURE   (judgement, UPPERCASE)
created_at: null
metric:
  primary: {name: avg_validation_R2_mean, value: null, direction: maximize}
---

# Trial <id> — <model_name> — <verdict>

<!-- ===== written BEFORE the run (agent) ===== -->

## Hypothesis
This is the first trial of the real HP-optimization campaign (as opposed to the prior
pipeline-validation smoke test, which used `n_epochs=5` and produced no scientifically
meaningful metric). No champion exists yet under this campaign's ledger, so this trial
makes no deliberate HP change: it simply re-runs the smoke test's best-performing HP
combination (lr=5e-4, weight_decay=0.0, dropout_rate=0.0, noise_std=0.0, patience=30) at
`n_epochs=200`, giving the model a real budget to converge (with early stopping on
`n_val=20`). I expect avg_validation_R2_mean to be substantially higher than the smoke
test's 0.5212, since 5 epochs could not have converged. This run establishes the
reference point that all subsequent deliberate HP changes in this campaign will be
measured against.

## Implementation
No architectural or hyperparameter change relative to the current `configs/autoencoder.yaml`
(this file is committed unmodified). Confirms config prerequisites are already correctly
set for this campaign: `hyper_automatic_values: false`, `multiple_models_and_dims: false`,
`n_val: 20` with `compute_metrics: true` (so validation_R2_mean/MSE_mean are logged),
`seed: 0` (reproducible).

<!-- ===== written AFTER the run ===== -->

## Results
<!-- Filled automatically by the driver — leave empty. It writes, for a completed trial:
     per-run metric values (by repeat axis), the aggregated primary metric,
     delta_vs_champion (display only), the also_log means, and the MLflow run ids.
     For a mechanically failed trial it writes the failure reason instead. -->

## Training Dynamics
<!-- Agent, after the run: stability, convergence speed, spikes, plateau, early stopping. -->

## Conclusion
<!-- Agent, after the run: did the hypothesis hold? Mechanistic explanation of why it
     worked or failed — not just the numbers. -->