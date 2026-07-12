---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2
summary: "Raise lr 10x (1e-5 -> 1e-4) after the baseline's scheduler decayed lr 4x before patience expired, suggesting 1e-5 was too conservative for this bottleneck size"
parent: 20fa5d8e

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
In the baseline (20fa5d8e, lr=1e-5), the best epoch was 52/200 and the plateau
scheduler (patience//5=6) had already halved lr four times (down to 6.25e-07) before
the 30-epoch early-stopping patience expired at epoch 82 — the last ~25 epochs of
budget were wasted on an effectively frozen model. This indicates lr=1e-5 is overly
conservative for this architecture at latent_dim=60: convergence was slow enough that
the scheduler ran out of room to make meaningful reductions before stalling. I
increase lr by 10x to 1e-4 (still 8x below the dim=240 champion's 8e-4, a deliberately
intermediate step given batch_size=1 makes higher lr riskier for stability) to reach a
better optimum in the same epoch budget, predicting an increase in
avg_validation_R2_mean.

## Implementation
Single-field change in configs/autoencoder.yaml: lr 1e-5 -> 1e-4. All other opened
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
