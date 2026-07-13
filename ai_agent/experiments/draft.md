---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2
summary: "Fine-tune lr under the final schedule — test 7e-4 (intermediate between the champion's 6e-4 and the failing 8e-4) under patience=60, a gap never tested since lr's optimum was originally found under patience=30"
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
lr=6e-4 was found optimal under patience=30 (patience_scheduler=6); lr=8e-4 was
re-confirmed to fail under patience=60 (patience_scheduler=12). But the intermediate
value, 7e-4, has never been tested under either schedule — the original lr sweep
went 6e-4 (win) -> 8e-4 (fail) directly. Since patience=60 changes how long the model
sits at its initial lr before decaying (potentially shifting the stability ceiling),
I test 7e-4 to check whether the lr optimum under the final, longer schedule is
still exactly 6e-4, or sits slightly higher (7e-4) now that more of the training
budget is available to smooth out any extra noise from a marginally larger step
size. Given the noise floor, I expect this to likely fail or be indistinguishable
from the champion, but it closes the one remaining untested gap in the lr axis.

## Implementation
Single-field change in configs/autoencoder.yaml: lr 6e-4 -> 7e-4, on top of the
champion's noise_std=0.0001, patience=60. weight_decay=0, dropout_rate=0 unchanged.
No architecture change.

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
