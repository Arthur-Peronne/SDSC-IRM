---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2
summary: Test intermediate lr=7e-4 with the champion's dropout=0.05 — a combination not yet tried
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
All lr values tested so far with `dropout=0.05` are 8e-4 (champion), 1e-3 (worse,
trial 8698c6f3). The old baseline's lr=5e-4 was never tested WITH dropout=0.05 either
(only alone, pre-dropout). `lr=7e-4` — between 5e-4 and 8e-4 — fills this gap: if the
dropout+lr interaction is a genuine, moderately wide effect (as trial 57e78778's
plateau finding suggests for dropout), a nearby lr should also land in the good
region rather than falling off sharply, helping distinguish "lr=8e-4 specifically
matters" from "any lr in the upper-mid range works about as well with dropout=0.05".

## Implementation
Single-field change relative to the champion: `lr: 8e-4 -> 7e-4`. `dropout_rate=0.05`
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