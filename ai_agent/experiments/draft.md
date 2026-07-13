---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2
summary: "Interaction re-check — retest dropout=0.05 (FAILED under patience=30 with a tight train/val gap of 0.045) now under the champion's patience=60, where the gap has widened back to 0.080"
parent: bed745a0

# ---- driver-written (leave null; the driver overwrites at lock/result) ----
id: null
status: draft
verdict: null
created_at: null
metric:
  primary: {name: avg_validation_R2_mean, value: null, direction: maximize}
---

# Trial <id> — <model_name> — <verdict>

<!-- ===== written BEFORE the run (agent) ===== -->

## Hypothesis
Dropout=0.05 FAILED earlier (-0.0151) because the train/val gap at that point (0.045,
under patience=30) was already tight — there was no overfitting left for dropout to
usefully correct, only capacity to lose. Since then, patience=60's longer schedule
reopened a real gap (train R2 0.8948 vs val 0.8148, gap 0.080, nearly double). This
changes the premise: dropout might now have genuine overfitting to correct rather
than just removing needed capacity. I predict this is more likely to help than the
first attempt, though not certain to beat the champion outright — it directly tests
whether the earlier FAILURE was conditional on the (now-changed) gap size, or an
unconditional property of this architecture/data at latent_dim=60.

## Implementation
Single-field change in configs/autoencoder.yaml: dropout_rate 0.0 -> 0.05, on top of
the champion's lr=6e-4, noise_std=0.0001, patience=60. weight_decay=0 unchanged. No
architecture change.

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
