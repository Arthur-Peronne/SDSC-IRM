---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2
summary: "Fifth replicate of patience=45, bringing its sample to parity (n=5) with patience=60, for the final variance comparison in the campaign report"
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
patience=45 has 4 points (0.8089, 0.8060, 0.7913, 0.7954; std≈0.0073); patience=60
has 5 (std≈0.0346). A 5th patience=45 draw puts both configs on equal sample size for
the final report's headline variance comparison. Given the tempering seen with
patience=49's 3rd point, I hold this prediction loosely: the tight cluster could
still widen, but even a moderate widening would likely leave patience=45
meaningfully tighter than patience=60/50's ~0.035 spread.

## Implementation
No change to configs/autoencoder.yaml — same config as 69832c74/59547f27/3fbb7661/
10c18786 (lr=6e-4, noise_std=0.0001, patience=45). No architecture change. Fifth
replicate of the patience=45 config.

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
