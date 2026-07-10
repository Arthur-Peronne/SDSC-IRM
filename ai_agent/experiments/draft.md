---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2
summary: Retry denoising at a much smaller magnitude (noise_std 0.0 -> 0.0002, ~10x smaller than trial 94b2bb2c's catastrophic 0.002)
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
Trial 94b2bb2c (`noise_std=0.002`, an Optuna-tuned value borrowed from a different
setting) failed catastrophically (delta -0.185): best_epoch collapsed to 9, train R2
itself dropped to 0.65, and validation std ballooned — the corruption dominated the
training signal, worsened by `batch_size=1` giving no batch-averaging to cancel the
per-sample noise. That trial does not test whether denoising helps at all — it tested
one badly-miscalibrated magnitude. I now use `noise_std=0.0002`, an order of magnitude
smaller, on the reasoning that at this scale the corruption should sit well below the
signal that drives the ~0.000846 train_loss floor seen in the baseline (loss values
are on the 1e-4..1e-3 scale, so this noise level should perturb inputs mildly rather
than dominate reconstruction). Given trial d78769c1's finding of a ~0.03 R2 run-to-run
noise floor, I only treat this as informative if the result clearly departs from that
band (either a real improvement above ~0.84, or another sharp collapse like trial
94b2bb2c's, rather than a marginal +/-0.03 wobble).

## Implementation
Single-field change in `configs/autoencoder.yaml`: `noise_std: 0.0 -> 0.0002`. All
other hyperparameters unchanged from the baseline (lr=5e-4, weight_decay=0.0,
dropout_rate=0.0, patience=30). No architectural change.

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