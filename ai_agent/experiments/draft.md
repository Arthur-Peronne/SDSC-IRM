---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2
summary: "Third control replicate of the champion (lr=8e-4, dropout=0.05), with remaining campaign budget dedicated to a robust confidence estimate"
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
With trials 04d1d849 through 1a7b6adf, the dropout/lr/weight_decay/noise_std/patience
combination space around the champion is now thoroughly mapped: no single-axis or
combined variation has exceeded 0.828, and two independent replicates of the exact
champion config landed at 0.803/0.805 (below even the old baseline). With 5 trials
of budget left and diminishing value in yet more new HP points (per trial 1a7b6adf's
conclusion), the most useful remaining use of the campaign is building a firmer
estimate of this configuration's typical performance for the final summary. This is
a third exact replicate, not a new idea — deliberate in the sense of directly
serving the campaign's final, honest conclusion about the champion's expected value.

## Implementation
No change to `configs/autoencoder.yaml` relative to the champion (`lr=8e-4`,
`dropout_rate=0.05`, all other fields at baseline defaults, `seed=0` untouched).

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