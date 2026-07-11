---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2
summary: Test patience increase (30 -> 45) on top of the champion's lr=8e-4 + dropout=0.05, the last opened HP not yet combined
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
`patience` is the one opened HP not yet tested in combination with the champion's
`lr=8e-4` + `dropout=0.05`. Under this combination, training now regularly runs
80-133 epochs (vs the old baseline's 66) before the 30-epoch patience window
triggers — the champion's own best epoch (80) is later than in most other trials,
suggesting the productive-training window has genuinely shifted with this
combination. A longer patience (45) gives the scheduler more room to anneal lr
further and potentially find a still-better epoch, similar in spirit to trial
d78769c1's patience test on the old baseline (which found no benefit there) — but
this is a different regime (regularized, slower-plateauing), so the old null result
does not necessarily transfer.

## Implementation
Single-field change relative to the champion: `patience: 30 -> 45`. `lr=8e-4` and
`dropout_rate=0.05` (the champion's values) unchanged. `weight_decay`, `noise_std`
unchanged from the baseline defaults. No architectural change.

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