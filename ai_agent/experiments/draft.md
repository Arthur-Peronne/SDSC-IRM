---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2
summary: "Switch axis to dropout (0.0 -> 0.05) on top of the champion's lr=6e-4, testing the dim=240 champion's dropout value now that the lr axis is closed"
parent: 97d513a3

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
The lr axis is closed (champion 97d513a3 at lr=6e-4, R2=0.7972; 8e-4 FAILED). I now
switch to dropout_rate, the next opened HP not yet touched since the neutral
baseline. I test 0.05 — the exact value used by the dim=240 campaign's champion — as
a natural reference point. However, the champion's train/val gap here is already
fairly tight (train R2 0.8422 vs val 0.7972, gap 0.045) compared to what justified
dropout at dim=240 (baseline gap there was much larger). I predict this is more
likely to hurt than help: dropout removes usable capacity from a model that, at this
4x-smaller bottleneck, may need all of it, and the gap is no longer the wide-overfit
profile that dropout is meant to correct. Testing it once cleanly establishes whether
regularization has any role at this operating point, or whether (as the FAILURE-heavy
back half of the dim=240 campaign around its own champion suggested) the found
optimum is already regularization-neutral.

## Implementation
Single-field change in configs/autoencoder.yaml: dropout_rate 0.0 -> 0.05, on top of
the champion's lr=6e-4. weight_decay=0, noise_std=0, patience=30 unchanged. No
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
