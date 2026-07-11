---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2
summary: Replicate dropout=0.04 (trial 57e78778) at lr=8e-4 to check whether that near-tying point is more robust than the champion's exact 0.05
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
Trial 57e78778 (`dropout=0.04`) scored 0.820 — near-tying the champion and beating
both exact-champion replicates (0.803, 0.805). Given the established ~0.025-0.03
noise floor, a single sample cannot say whether 0.04 is itself a favorable draw (like
the champion's original 0.828) or a genuinely more typical/robust point. A replicate
at 0.04 directly tests this: if it lands close to 0.82 again, dropout=0.04 may be a
more reliably good setting than 0.05; if it drops toward 0.75-0.80 like other single
"off" draws, 0.04 is likely just as noisy as everywhere else in the plateau.

## Implementation
No new HP value — replicate of trial 57e78778's config relative to the champion:
`dropout_rate: 0.05 -> 0.04`, `lr=8e-4` unchanged. `weight_decay`, `noise_std`,
`patience` unchanged from the baseline defaults.

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