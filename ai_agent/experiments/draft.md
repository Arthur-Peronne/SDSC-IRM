---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2
summary: "Third replicate of patience=49 (0.8074, 0.8082, ?) for a final, more confident variance estimate before closing out the campaign's schedule-length investigation"
parent: bc589070

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
The first two patience=49 draws (0.8074, 0.8082) were remarkably tight (spread
0.0008). A third draw will confirm whether this extreme tightness holds or was
itself a 2-point coincidence; either way, 3 points at patience=49 plus 4 at
patience=45 (both scheduler=9) versus 5 at patience=60 and 2 at patience=50
(scheduler=12 and 10) gives a reasonably powered comparison to close this line of
investigation before wrapping up the campaign.

## Implementation
No change to configs/autoencoder.yaml — same config as 03af338b/a238f026 (lr=6e-4,
noise_std=0.0001, patience=49, patience_scheduler=9). No architecture change. Third
replicate of the patience=49 config.

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
