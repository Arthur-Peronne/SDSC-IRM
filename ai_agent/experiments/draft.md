---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2
summary: Fine-tune dropout at 0.045 (between the near-tying 0.04 and the champion's 0.05), resuming deliberate HP exploration under the extended max_trials=50 budget
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
The campaign budget was extended by the user from 30 to 50 trials (committed
directly to experiment.yaml, outside this trial). With the noise floor now well
characterized (~0.03-0.04 R2, confirmed by 3 champion replicates clustering around
0.78-0.83 rather than the recorded 0.828), further single points are unlikely to be
individually conclusive, but the extra budget allows denser sampling of the good
dropout region to build a clearer overall shape. `dropout_rate=0.045` sits between
the near-tying 0.04 (trials 57e78778/3819dd6a: 0.820/0.784) and the champion's 0.05
(0.828/0.803/0.805/0.780) — filling a gap in the resolution of this plateau.

## Implementation
Single-field change relative to the champion: `dropout_rate: 0.05 -> 0.045`. `lr=8e-4`
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