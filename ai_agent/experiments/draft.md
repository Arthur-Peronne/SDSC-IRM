---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2
summary: Push lr further (5e-4 -> 1e-3) after trial f23a08a7's lr=8e-4 nearly tied the champion with a better raw val_loss
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
Trial f23a08a7 (`lr=8e-4`) essentially tied the champion (delta -0.0009, inside the
noise band) while reaching the lowest raw `val_loss` of the campaign (0.000548) and
converging faster (best_epoch=48 vs 66) — no instability. This is the strongest lead
so far, so I continue in the same direction rather than switching axes again:
`lr=1e-3`, twice the baseline. This is still moderate compared to the smoke test's
1.5e-3 (which failed, but under `n_epochs=5` — a truncated, uninformative regime for
this question). If the "more productive exploration at higher lr" mechanism keeps
holding, this could cross from a tie into an actual improvement; if lr is approaching
an instability threshold, this trial should reveal it (noisier val curve, worse final
R2) before going further.

## Implementation
Single-field change in `configs/autoencoder.yaml`: `lr: 5e-4 -> 1e-3`. All other
hyperparameters unchanged from the baseline (weight_decay=0.0, dropout_rate=0.0,
noise_std=0.0, patience=30). No architectural change.

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