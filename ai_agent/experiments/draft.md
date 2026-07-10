---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2
summary: Test lr increase (5e-4 -> 8e-4), the untested direction, after trial e44c7fa6 found lowering lr hurt generalization
parent: 185cf97f

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
Trial e44c7fa6 (`lr=2e-4`) converged to its best epoch much sooner (43 vs the
baseline's 66) and landed in a worse-generalizing optimum (val_R2 0.789 vs 0.807),
suggesting that the baseline's higher lr benefits from "exploring longer" before
settling rather than being "too coarse". This trial tests the untested direction:
`lr=8e-4`, a moderate increase (not as aggressive as the smoke-test's 1.5e-3, which
failed under an unrelated n_epochs=5 truncation and is not informative here). If the
exploration-length mechanism is right, a higher lr should explore even more before
the scheduler anneals it down on plateau, potentially reaching a better-generalizing
basin than the baseline. Given the ~0.03 R2 noise floor (trial d78769c1), I only treat
this as a real effect if it clearly exceeds that band.

## Implementation
Single-field change in `configs/autoencoder.yaml`: `lr: 5e-4 -> 8e-4`. All other
hyperparameters unchanged from the baseline (weight_decay=0.0, dropout_rate=0.0,
noise_std=0.0, patience=30). No architectural change.

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