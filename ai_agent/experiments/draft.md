---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2
summary: "Ablation — remove noise_std (0.0001 -> 0.0) from the champion under patience=60, to check whether its small individual win (found under patience=30) still holds at the much longer schedule"
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
noise_std=0.0001 was found to help (+0.0027) under patience=30, and the champion
bed745a0 carries it forward combined with patience=60 — but that combination was
never isolated: patience was tuned WITH noise_std=0.0001 already active, so its
marginal contribution at the new, much longer schedule was never re-verified. Unlike
the weight-space regularizers (dropout, weight_decay), noise_std is a light,
capacity-neutral input augmentation, so I predict its small benefit persists or even
grows slightly with more epochs (more exposure to the augmentation), i.e. removing it
(this trial) should produce a small FAILURE relative to the champion, confirming
noise_std still contributes positively rather than having been a fragile,
schedule-specific artifact.

## Implementation
Single-field change in configs/autoencoder.yaml: noise_std 0.0001 -> 0.0 (ablation),
on top of the champion's lr=6e-4, patience=60. weight_decay=0, dropout_rate=0
unchanged. No architecture change.

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
