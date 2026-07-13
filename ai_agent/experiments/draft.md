---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2      # the AE architecture trained (e.g. AE3dDilatedAttention). Descriptive, NOT the id.
summary: "Baseline at latent_dim=8: dim=60 champion HPs unchanged (lr=6e-4, weight_decay=0, dropout_rate=0, noise_std=0.0001, patience=50)"         # one-line description of the change (becomes the CSV modification_description)
parent: null          # lineage: `id` of the trial whose CODE this one branched FROM. null for roots.

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
This is the root trial of the dim=8 campaign (fresh ledger, no champion yet). I am
carrying over the HP set found optimal at latent_dim=60
(`ai_agent/experiments/aiagent_HP_sepv2_60/REPORT_dim60.md`: lr=6e-4, weight_decay=0,
dropout_rate=0, noise_std=0.0001, patience=50) unchanged, to measure how that
optimum transfers to a ~7.5x smaller bottleneck. I do not assume it will hold as-is:
a dim=8 bottleneck is already a much stronger capacity constraint, so the
regularization trade-off (e.g. dropout, noise) may shift — this run's only purpose
is to fix that anchor point before making any deliberate HP change.

## Implementation
No hyperparameter change relative to the currently committed
`configs/autoencoder.yaml` (already set to latent_dimensions=8, lr=6e-4,
weight_decay=0.0, dropout_rate=0.0, noise_std=0.0001, patience=50). This trial
simply logs `avg_validation_R2_mean` for that fixed config under latent_dim=8, to
serve as the BASELINE row of the new campaign ledger.

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