---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2
summary: Lower lr (5e-4 -> 2e-4) for finer convergence, switching axis after two regularization FAILUREs (dropout, weight_decay)
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
Trials 6a8a38d6 (dropout=0.15) and 5fb49bb5 (weight_decay=1e-5) both FAILED by
*removing* model capacity, worse than the baseline. That rules out the "excess
capacity / overfitting" framing and points instead at the optimization trajectory:
the baseline's val_loss reaches its minimum quickly (epoch 66) then plateaus/creeps
up rather than continuing to improve, which is also consistent with the LR (starting
at 5e-4) being too coarse to keep refining the minimum once the model is in its
basin — the ReduceLROnPlateau-style scheduler only starts decaying *after* patience
epochs without improvement, by which point the run is already coasting toward early
stop. I set `lr: 5e-4 -> 2e-4` so the model takes smaller, more precise steps from
the start, predicting it reaches a better (lower) val_loss minimum, even if it takes
somewhat more epochs to get there (n_epochs=200 budget and patience=30 unchanged, so
there is room). I predict `avg_validation_R2_mean` improves above 0.8075.

## Implementation
Single-field change in `configs/autoencoder.yaml`: `lr: 5e-4 -> 2e-4`. All other
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