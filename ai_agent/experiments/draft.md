---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2
summary: Replicate lr=6e-4 + dropout=0.05 (trial 1f00c6a2, 0.802) to check whether this alternative point is more stable/robust than the fragile champion
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
The campaign found the champion (`lr=8e-4`+`dropout=0.05`) sits at a narrow, fragile
peak (lr=9e-4 already fails badly; 3 champion replicates spread 0.78-0.83). By
contrast, `lr=6e-4`+`dropout=0.05` (trial 1f00c6a2, 0.802) sits in a wider, more
lr-tolerant region (5e-4, 6e-4, 8e-4 all work; only the isolated 7e-4 valley fails).
A replicate here tests whether this alternative point is ALSO more stable run-to-run
(smaller variance), which would make it a better practical recommendation than the
champion despite its slightly lower single-run score, or whether it is equally noisy
and the "wider lr tolerance" doesn't imply "lower run-to-run variance".

## Implementation
No new HP value — replicate of trial 1f00c6a2's config relative to the champion:
`lr: 8e-4 -> 6e-4`, `dropout_rate=0.05` unchanged. `weight_decay`, `noise_std`,
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