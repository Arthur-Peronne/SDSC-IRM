---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2
summary: Test dropout=0.06 at lr=5e-4 — same transfer question as trial 9a54d15d, at the dropout value closer to the champion's 0.05
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
Trial 9a54d15d found `dropout=0.08` at `lr=5e-4` does not fail the way it does at
`lr=8e-4` (0.798 vs 0.756, both individually within/near noise but 5e-4's version
clearly not unstable). This tests `dropout=0.06` at `lr=5e-4` — the value that,
at lr=8e-4, gave a real, clear degradation (0.795, delta -0.033, non-noise). If
`lr=5e-4` similarly "absorbs" this dropout value without the degradation seen at
8e-4, it reinforces that the dropout/lr interaction — not either parameter alone —
governs where the good regions are, and that lr=5e-4 may have a wider dropout
tolerance than lr=8e-4 generally.

## Implementation
Two-field change relative to the champion: `lr: 8e-4 -> 5e-4` AND
`dropout_rate: 0.05 -> 0.06`. `weight_decay`, `noise_std`, `patience` unchanged from
the baseline defaults. No architectural change.

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