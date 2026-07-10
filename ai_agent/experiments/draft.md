---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2      # the AE architecture trained (e.g. AE3dDilatedAttention). Descriptive, NOT the id.
summary: "Baseline HPs as currently configured (lr=5e-5, weight_decay=0.0, dropout_rate=0.0, noise_std=0.0, patience=30); procedural smoke test (n_epochs=5, not scientifically meaningful)"
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
<!-- Filled automatically by the driver — leave empty. It writes, for a completed trial:
     per-run metric values (by repeat axis), the aggregated primary metric,
     delta_vs_champion (display only), the also_log means, and the MLflow run ids.
     For a mechanically failed trial it writes the failure reason instead. -->

## Training Dynamics
<!-- Agent, after the run: stability, convergence speed, spikes, plateau, early stopping. -->

## Conclusion
<!-- Agent, after the run: did the hypothesis hold? Mechanistic explanation of why it
     worked or failed — not just the numbers. -->