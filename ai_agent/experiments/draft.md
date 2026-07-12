---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2
summary: "Push lr further (3e-4 -> 6e-4), close to the dim=240 champion's 8e-4, since the last two lr increases both won with diminishing but still-positive gains and no instability yet"
parent: 2b46a997

# ---- driver-written (leave null; the driver overwrites at lock/result) ----
id: null
status: draft
verdict: null
created_at: null
metric:
  primary: {name: avg_validation_R2_mean, value: null, direction: maximize}
---

# Trial <id> — <model_name> — <verdict>

<!-- ===== written BEFORE the run (agent) ===== -->

## Hypothesis
Two consecutive lr increases (1e-5->1e-4, then ->3e-4) both improved
avg_validation_R2_mean with no instability, and additionally narrowed the train/val
gap and lowered validation R2 std each time — evidence that higher lr is acting as an
implicit regularizer here, not just speeding up convergence. The gains are
diminishing (+0.096, then +0.040), which could mean either a genuine approach to a
local optimum, or that we are still on the same monotonic trend. I push once more to
6e-4 (close to, but still below, the dim=240 champion's 8e-4) to distinguish the two:
a further (even if small) improvement keeps the axis open toward 8e-4; a plateau or
regression, or the first sign of instability (val loss spikes, NaNs), would mark 3e-4
as the practical optimum for this batch_size=1 setup and shift exploration to the
other 4 opened HPs.

## Implementation
Single-field change in configs/autoencoder.yaml: lr 3e-4 -> 6e-4. All other opened
HPs unchanged (weight_decay=0, dropout_rate=0, noise_std=0, patience=30). No
architecture change.

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
