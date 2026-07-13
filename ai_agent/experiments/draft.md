---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2
summary: "Test patience=49 (patience_scheduler=9, same as 45) vs patience=50 (scheduler=10) to isolate whether patience=45's tight variance is driven by the scheduler step or by patience itself"
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
patience=45 (patience_scheduler=45//5=9) shows tight variance (n=4, std≈0.007);
patience=50 (patience_scheduler=50//5=10) already shows wide variance at n=2
(spread 0.035), matching patience=60's scale. Since patience_scheduler jumps from 9
to 10 exactly at this boundary, I test patience=49 — same patience_scheduler=9 as
45 (49//5=9), but 4 epochs longer overall patience — to isolate the variable: if
patience=49 is ALSO tight (like 45), the scheduler value (how often lr decays) is
the real driver of reproducibility, not patience/early-stopping length itself; if
patience=49 is wide (like 50), then the earlier finding was more likely a
coincidence tied specifically to the exact value 45, not a mechanistic
scheduler-step effect.

## Implementation
Single-field change in configs/autoencoder.yaml: patience 50 -> 49 (patience_scheduler
auto-derived as patience//5 = 9, matching patience=45's scheduler value), on top of
lr=6e-4, noise_std=0.0001. weight_decay=0, dropout_rate=0 unchanged. No architecture
change.

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
