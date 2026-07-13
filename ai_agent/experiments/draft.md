---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2
summary: "Map the noise_std axis below its current value — test 0.00005 (half of the champion's 0.0001), the one point on this axis never tried, now interpreting the result against the ~0.03 noise floor just established"
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
The noise_std axis was mapped at 0.0, 0.0001 (win), 0.0002 (worse), 0.0003 (worse) —
but never below 0.0001. Given the ablation (0.0) already showed 0.0001 contributes
positively, and the sharp non-monotonic cliff appears only ABOVE 0.0001, I test
0.00005 to check whether the true optimum on this axis is even smaller than 0.0001,
or whether 0.0001 is already at/near the peak. Given the just-established ~0.03
single-run noise floor, and this axis's individual effect sizes being small (~0.003-
0.01), I do not expect this result to be conclusively distinguishable from noise
either way — but a result far outside the noise band in either direction would still
be informative, and a within-noise result would confirm that 0.0001 is "good enough"
and further fine-tuning of this axis is not worth additional trials.

## Implementation
Single-field change in configs/autoencoder.yaml: noise_std 0.0001 -> 0.00005, on top
of the champion's lr=6e-4, patience=60. weight_decay=0, dropout_rate=0 unchanged. No
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
