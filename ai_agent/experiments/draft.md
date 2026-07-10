---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2
summary: Increase patience (30 -> 50), the last untried opened HP, to test whether the plateau after epoch 66 ever recovers given more room
parent: 185cf97f

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
This completes the first single-axis pass around the baseline: `patience` is the last
of the 5 opened hyperparameters not yet tried in isolation. In the baseline, `val_loss`
plateaued/crept slightly upward for the full 30-epoch patience window after its
minimum at epoch 66, with no sign of a dip back down — so I do not expect a dramatic
change. But the LR scheduler keeps annealing lr on every plateau epoch (down to ~1e-6
range by the time patience triggers in other trials), so a longer patience window
gives the scheduler more room to reach an even smaller lr before stopping, which could
in principle let the model settle slightly further. I set `patience: 30 -> 50` to test
this directly rather than assume. I predict the result is close to a wash (aggregate
near 0.8075) rather than a large change in either direction — a genuinely uncertain,
informative test rather than an expected win.

## Implementation
Single-field change in `configs/autoencoder.yaml`: `patience: 30 -> 50`. All other
hyperparameters unchanged from the baseline (lr=5e-4, weight_decay=0.0,
dropout_rate=0.0, noise_std=0.0). No architectural change.

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