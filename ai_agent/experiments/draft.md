---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2
summary: "Push lr further (1e-4 -> 3e-4) after the previous trial's clean win, to test whether the improving trend continues or 1e-4 was already near the local optimum"
parent: 47e474ef

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
The champion 47e474ef (lr=1e-4) converged cleanly and fast — no instability, best
epoch at 37/200, val R2 std improved over the baseline — with room left below the
dim=240 champion's lr=8e-4. Since raising lr 10x from a clearly-too-conservative
1e-5 gave a strong, stable improvement, I test whether the trend continues with a
further 3x increase (1e-4 -> 3e-4), predicting either a further increase in
avg_validation_R2_mean (if 1e-4 was still leaving convergence speed on the table) or
a plateau/regression (if 1e-4 is already close to the batch_size=1 stability ceiling
for this architecture) — either outcome narrows the useful lr range for the next
trials.

## Implementation
Single-field change in configs/autoencoder.yaml: lr 1e-4 -> 3e-4. All other opened
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
