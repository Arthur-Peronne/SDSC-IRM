---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2      # the AE architecture trained (e.g. AE3dDilatedAttention). Descriptive, NOT the id.
summary: "Campaign baseline: AE3dAsymResSeparableV2, lr=1e-4, patience=20, weight_decay=dropout_rate=noise_std=0"         # one-line description of the change (becomes the CSV modification_description)
parent: null         # lineage: `id` of the trial whose CODE this one branched FROM. null for roots.

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
This is the opening trial of a new campaign (fresh ledger, tag `AIagent_classification_testClaude`),
not a comparison against a prior trial in this ledger. `AE3dAsymResSeparableV2` was the strongest
architecture from the previous (archived) campaign on reconstruction R2, so it is used here as the
starting point. Regularization (`weight_decay`, `dropout_rate`, `noise_std`) is set to 0 and `lr`
lowered to 1e-4 (from the previous campaign's 8e-4) to establish an unregularized, conservative-lr
reference point for this campaign's classification-accuracy metric, before any deliberate
architecture/HP exploration begins.

## Implementation
No code change to `src/models/ae_models.py` — `AE3dAsymResSeparableV2` (and its `build_autoencoder`
branch) already exist. Only `configs/autoencoder.yaml` hyperparameters changed:
- `lr`: 8e-4 -> 1e-4
- `weight_decay`: 1e-5 -> 0.0
- `dropout_rate`: 0.3 -> 0.0
- `noise_std`: unchanged at 0.0
- `patience`: unchanged at 20
- `model_name` / `latent_dimensions`: unchanged (`AE3dAsymResSeparableV2`, 20)

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