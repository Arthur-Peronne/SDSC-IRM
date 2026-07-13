---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2      # the AE architecture trained (e.g. AE3dDilatedAttention). Descriptive, NOT the id.
summary: "lr=7e-4 replicate 2/6: building a matched sample for this near-tied candidate (n=1 so far: 0.7999) against the champion's n=6"         # one-line description of the change (becomes the CSV modification_description)
parent: 02343abb          # lineage: `id` of the trial whose CODE this one branched FROM. null for roots.

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
The noise_std comparison (now closed as a clean tie at matched n=6) validates
this replication strategy. Applying the same approach to lr=7e-4, the other
near-tied single-run candidate (trial 95ede5a9: 0.7999, delta -0.0056): building
it to a matched n=6 to see whether its true mean is competitive with, better
than, or worse than the champion's 0.7817, rather than relying on one ambiguous
data point.

## Implementation
`configs/autoencoder.yaml`: `lr: 6e-4 -> 7e-4`. Everything else unchanged:
weight_decay=0.0, dropout_rate=0.0, noise_std=0.0, patience=50,
latent_dimensions=8, seed=0.

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