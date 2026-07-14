---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2      # the AE architecture trained (e.g. AE3dDilatedAttention). Descriptive, NOT the id.
summary: "Sanity check: patience=45 under lr=7e-4 — does the patience=50 recommendation (established under lr=6e-4) still hold?"         # one-line description of the change (becomes the CSV modification_description)
parent: 71508734          # lineage: `id` of the trial whose CODE this one branched FROM. null for roots.

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
Same rationale as the dropout sanity check: patience=50 vs 45/70/30 was
characterized under lr=6e-4. With lr=7e-4 now the recommended point and its own
convergence dynamics (best epochs 43-76, LR schedule interacting with training
speed), the patience_scheduler cadence effect could plausibly shift. Quick
single-run check with the last few trials of budget.

## Implementation
`configs/autoencoder.yaml`: `lr: 7.5e-4 -> 7e-4`, `patience: 50 -> 45`.
Everything else unchanged: weight_decay=0.0, dropout_rate=0.0, noise_std=0.0,
latent_dimensions=8.

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