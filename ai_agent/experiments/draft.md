---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2
summary: Fine-tune dropout downward (0.05 -> 0.04) at lr=8e-4, closer than the failed 0.03, for a finer-resolution symmetric check
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
The dropout grid so far (0.03, 0.05=champion, 0.06, 0.08) showed 0.05 winning against
all neighbors, but the noise-characterization trials (59ff727f, f5873a7c) revealed
the champion's own config varies by ~0.025 R2 run-to-run — comparable to the gap seen
at 0.06 (-0.033). This makes it worth checking `dropout_rate=0.04`, closer to the
champion than 0.03 was, both to test the other side symmetrically at fine resolution
and to see whether a result lands within the now-established noise band (suggesting
0.04-0.06 are all statistically similar to 0.05) or still shows a real drop (0.03-like)
that would support a genuinely narrow optimum rather than a noisy plateau.

## Implementation
Single-field change relative to the champion: `dropout_rate: 0.05 -> 0.04`. `lr=8e-4`
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