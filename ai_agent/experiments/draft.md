---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2
summary: "Test a small weight_decay (0.0 -> 1e-6) on top of the champion's lr=6e-4, a gentler/smoother regularizer than the dropout that just FAILED"
parent: 97d513a3

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
Dropout (0.05) just FAILED on top of the champion (97d513a3, lr=6e-4): it removed
usable capacity the model needed, since the train/val gap there (0.045) was already
tight, not a wide-overfit profile. Unlike dropout, weight_decay is a smooth,
deterministic L2 penalty with no added stochastic noise, so it should not compound
with the raw-lr noise already present at 6e-4, and it shrinks weights globally rather
than randomly zeroing activations — a softer intervention. I test a very small value
(1e-6) to see if this gentler regularizer can still nudge val R2 up without the
capacity loss dropout caused. Given the tight gap, I still predict a likely FAILURE
(mirroring the dim=240 campaign, where an equally small weight_decay=1e-6 on top of
ITS champion also failed) but a milder one than dropout's -0.0151.

## Implementation
Single-field change in configs/autoencoder.yaml: weight_decay 0.0 -> 1e-6, on top of
the champion's lr=6e-4. dropout_rate=0.0 (reverted), noise_std=0, patience=30
unchanged. No architecture change.

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
