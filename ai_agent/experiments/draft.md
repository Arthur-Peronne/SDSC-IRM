---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2
summary: Test patience decrease (30 -> 15) on the champion's lr=8e-4 + dropout=0.05, completing the patience direction not yet tried
parent: 3e07b08d

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
Trial 571f33ab tested `patience=45` (increase) on the champion and found it neutral.
The decrease direction (`patience=15`) has not been tested at this operating point.
Across the champion-neighborhood trials, best epochs have ranged roughly 40-100+,
meaning a patience of only 15 risks stopping before the run reaches its true minimum
on an unlucky draw — but if the loss curve's early improvements are usually found
well within 15 epochs of no-improvement windows, a shorter patience could reach a
similar result faster (a practical/efficiency question) without hurting the metric.
I predict this either ties the champion-neighborhood's typical performance (if 15 is
enough runway) or shows a clear regression (if useful improvements are often found
later than a 15-epoch plateau would tolerate).

## Implementation
Single-field change relative to the champion: `patience: 30 -> 15`. `lr=8e-4` and
`dropout_rate=0.05` unchanged. `weight_decay`, `noise_std` unchanged from the
baseline defaults. No architectural change.

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