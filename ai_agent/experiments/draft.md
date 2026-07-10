---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2      # the AE architecture trained (e.g. AE3dDilatedAttention). Descriptive, NOT the id.
summary: "lr 5e-5 -> 5e-4 (10x): faster per-epoch convergence under the fixed 5-epoch budget"
parent: 21acba9e      # lineage: `id` of the trial whose CODE this one branched FROM. null for roots.

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
In the BASELINE (21acba9e), val loss at all 3 latent dims was still descending steeply
at epoch 5/5 (e.g. dim=8: 0.072 -> 0.0029, still improving, no plateau) — the campaign's
fixed `n_epochs: 5` cuts training off long before convergence, not because the model has
converged. Under a hard epoch-budget constraint (n_epochs is frozen, not in the mutable
5), the direct lever to make more progress per epoch is a higher learning rate. I
increase `lr` from 5e-5 to 5e-4 (10x, one log-step), all else unchanged. I predict this
raises `avg_validation_R2_mean` above the BASELINE's -0.5428, because faster convergence
per step should let the model reach a lower loss / higher R² within the same 5-epoch
budget, mechanistically compensating for the truncated schedule (rather than fixing an
optimization pathology — none was observed in the BASELINE curves).

## Implementation
`configs/autoencoder.yaml`: `lr: 5e-5` -> `lr: 5e-4`. No other field touched
(`weight_decay`, `dropout_rate`, `noise_std`, `patience` unchanged from BASELINE;
architecture, data, split, `n_epochs` all frozen/untouched).

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