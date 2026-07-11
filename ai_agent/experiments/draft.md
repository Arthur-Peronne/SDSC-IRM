---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2
summary: Fine-tune dropout upward (0.05 -> 0.06) at lr=8e-4, between the champion and the failed 0.08
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
Trials 274ff200 (dropout=0.03) and 3faa0f66 (dropout=0.08) both showed a real,
non-noise degradation relative to the champion's dropout=0.05, unlike the noisier
lr neighborhood. This suggests a genuine, fairly sharp optimum near 0.05, worth
resolving at finer granularity than the 0.03/0.05/0.08 grid tested so far.
`dropout_rate=0.06` tests just above the champion, a small step (vs the 0.03 jump to
0.08), predicting a result close to the champion's if 0.05 is near the true optimum,
or a further improvement if the optimum lies slightly higher.

## Implementation
Single-field change relative to the champion: `dropout_rate: 0.05 -> 0.06`. `lr=8e-4`
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