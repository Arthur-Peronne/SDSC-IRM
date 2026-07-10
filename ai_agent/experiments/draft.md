---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2
summary: Add mild dropout (0.0 -> 0.15) to regularize against the train/val R2 gap seen in the baseline
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
In the baseline (185cf97f), `train_R2_mean` (0.859) exceeded `validation_R2_mean`
(0.807) by ~0.05, and `val_loss` plateaued/crept upward for 30 epochs after its
minimum (epoch 66) while `train_loss` kept falling — a classic mild-overfitting
signature, with both `dropout_rate` and `noise_std` currently at 0 (no regularization
active). I set `dropout_rate: 0.0 -> 0.15` because dropout should reduce
co-adaptation of the encoder/decoder units on the 100 training patients, narrowing
the train/val gap and letting validation R2 keep improving instead of plateauing
early. I predict `avg_validation_R2_mean` increases above 0.8075.

## Implementation
Single-field change in `configs/autoencoder.yaml`: `dropout_rate: 0.0 -> 0.15`. All
other hyperparameters unchanged from the baseline (lr=5e-4, weight_decay=0.0,
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