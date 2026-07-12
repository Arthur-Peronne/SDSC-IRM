---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2
summary: "Push noise_std further (0.0001 -> 0.0003) after the small win, to test whether the denoising axis has more headroom before approaching the magnitude that caused catastrophic failure at dim=240"
parent: 319dacea

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
noise_std=0.0001 was the only regularizer to help so far (+0.0027), unlike dropout
(hurt badly) and weight_decay (neutral), because it does not tax model capacity the
way weight-space regularizers do. I test a 3x increase (0.0003) to see if the
denoising axis behaves like the lr axis did — more of a good thing helping further,
up to a point — while staying a full order of magnitude below the 0.002 value that
caused a catastrophic R2 -0.185 collapse in the dim=240 campaign (at a much lower
lr=5e-4, where the relative perturbation-to-signal ratio may have been different
anyway). Predicting either a further small gain (axis has headroom) or a
plateau/regression (0.0001 was already the local optimum for this mechanism).

## Implementation
Single-field change in configs/autoencoder.yaml: noise_std 0.0001 -> 0.0003, on top
of the champion's lr=6e-4. weight_decay=0, dropout_rate=0, patience=30 unchanged. No
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
