---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2
summary: Stack weight_decay=1e-6 AND noise_std=0.0001 together (both individually neutral) on the champion's lr=8e-4 + dropout=0.05
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
Trials de0f5947 (`weight_decay=1e-6` alone on the champion) and 5033c15b
(`noise_std=0.0001` alone on the champion) were both individually neutral (deltas
-0.017 and -0.030, inside/at the edge of the noise band). Neither showed the
catastrophic effect their larger counterparts did (weight_decay=1e-5, noise_std=0.002),
so both are "safe" at this magnitude. This trial fuses them: stacking two mechanistically
different, individually-harmless regularizers (deterministic L2 + input-space
corruption) together, on top of the already-working dropout, to test whether
they compound (either positively, if they attack different aspects of the same
generalization gap, or negatively, if the "regularization budget" is a real
constraint and even harmless individual additions become harmful in combination).

## Implementation
Two-field addition relative to the champion: `weight_decay: 0.0 -> 1e-6` AND
`noise_std: 0.0 -> 0.0001`. `lr=8e-4` and `dropout_rate=0.05` unchanged. `patience`
unchanged from the baseline default. No architectural change.

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