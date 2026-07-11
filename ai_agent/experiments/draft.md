---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2
summary: Fill the lr gap between the champion (8e-4) and the failed 1e-3 — test lr=9e-4 + dropout=0.05
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
`lr=8e-4` (champion) works well with dropout=0.05; `lr=1e-3` fails via genuine
instability even with dropout=0.05 (trial 8698c6f3). `lr=9e-4` fills the untested
gap between them. Given the campaign's finding that this landscape has sharp,
localized features (the isolated 7e-4 valley) rather than smooth trends, this
trial's outcome is genuinely uncertain a priori: it could be as good as 8e-4, as bad
as 1e-3, or something in between — informative either way for bounding how narrow
the champion's good lr region really is.

## Implementation
Single-field change relative to the champion: `lr: 8e-4 -> 9e-4`. `dropout_rate=0.05`
unchanged. `weight_decay`, `noise_std`, `patience` unchanged from the baseline
defaults. No architectural change.

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