---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2
summary: "Map the patience boundary — test 75 (patience_scheduler=15, between the winning 60/12 and the failing 90/18) to refine the practical optimum of the schedule-length axis"
parent: bed745a0

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
patience=60 (patience_scheduler=12) is champion; patience=90 (patience_scheduler=18)
reversed the trend, worsened by late-stage overfitting in the frozen-lr tail. Unlike
the noise_std axis (a genuine sharp, non-monotonic cliff confirmed by a midpoint that
was worse than both neighbors), the patience axis's underlying mechanism (schedule
length) is smoother and directional, so I expect patience=75 (patience_scheduler=15)
to fall between the two: either a small further win over 60 (meaning 60 was not
quite the peak) or a small loss (confirming 60 as the peak with 75 already on the
downslope) — not another chaotic jump. This resolves whether 60 is exactly optimal or
merely close.

## Implementation
Single-field change in configs/autoencoder.yaml: patience 60 -> 75 (patience_scheduler
auto-derived as patience//5 = 15), on top of the champion's lr=6e-4, noise_std=0.0001.
weight_decay=0, dropout_rate=0 unchanged. No architecture change.

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
