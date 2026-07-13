---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2
summary: "Final campaign trial (budget limit) — lr=3e-4 (a gentler, previously-good lr) combined with weight_decay=1e-6, never tested together; closes the lr x weight_decay interaction gap"
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
weight_decay=1e-6 was tested only ever on top of lr=6e-4 (near-neutral to mildly
negative, -0.004 to -0.008 across patience contexts). lr=3e-4 was itself a genuine
CHAMPION early in the campaign (+0.040 over the 1e-4 baseline) before being
superseded by higher lr values. The two were never combined. Since weight_decay's
stabilizing effect on noise_std=0.0002 (trials 75165602/b93971e6) showed it CAN
meaningfully interact with other axes rather than acting purely independently, I
test whether weight_decay behaves differently at this gentler lr — e.g., a lower lr
already has less raw-gradient noise (as seen throughout the lr sweep), so
weight_decay's smoothing might be more additive/neutral here than at 6e-4, or
conversely the slower convergence at 3e-4 combined with weight_decay's extra pull
toward zero could compound into a larger relative cost. Given the campaign's overall
evidence, I predict a mild FAILURE (weight_decay has not helped in any tested
context), but this closes out the lr x weight_decay interaction gap as the
campaign's final trial.

## Implementation
Two-field change in configs/autoencoder.yaml: lr 6e-4 -> 3e-4 AND weight_decay 0.0
-> 1e-6, simultaneously, on top of the champion's noise_std=0.0001, patience=50
(patience left untouched). dropout_rate=0 unchanged. No architecture change. This is
the campaign's 40th and final trial (max_trials budget).

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
