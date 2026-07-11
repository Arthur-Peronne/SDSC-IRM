---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2
summary: Test whether dropout=0.05 stabilizes lr=1e-3, which failed alone (trial e8857e8d) due to instability
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
Trial e8857e8d (`lr=1e-3` alone) FAILED with a clear instability signature
(validation_R2_std jumped to 0.118, best val_loss worse than champion). Separately,
trial 3e07b08d showed that `dropout=0.05` combined with `lr=8e-4` reduced instability
markers (lower std, narrower train/val gap) relative to `lr=8e-4` alone. Fusing these:
if dropout's regularizing effect is what let `lr=8e-4` outperform its unregularized
counterpart, it may also tame the instability that made `lr=1e-3` fail alone, letting
this even-more-aggressive lr's faster/deeper exploration finally pay off instead of
destabilizing. I test `lr=1e-3` + `dropout=0.05` together, predicting either a further
improvement over the champion (if this reasoning holds) or a result resembling
e8857e8d's instability (elevated std, worse val_loss) if dropout=0.05 is not strong
enough to tame this particular lr.

## Implementation
Single-field change relative to the champion: `lr: 8e-4 -> 1e-3`. `dropout_rate=0.05`
(the champion's value) unchanged. `weight_decay`, `noise_std`, `patience` unchanged
from the baseline defaults. No architectural change.

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