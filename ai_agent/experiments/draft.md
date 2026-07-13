---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2      # the AE architecture trained (e.g. AE3dDilatedAttention). Descriptive, NOT the id.
summary: "Sanity check: dropout_rate=0.05 under lr=7e-4 (the new recommended lr) — does the 'dropout is harmful' conclusion (established under lr=6e-4) still hold?"         # one-line description of the change (becomes the CSV modification_description)
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
All the regularization-axis conclusions (dropout, weight_decay, noise_std all
harmful-or-neutral) were established while lr=6e-4. Now that lr=7e-4 is the
recommended point, I want a quick single-run sanity check that dropout is still
clearly harmful in this new context, rather than assuming it transfers
unquestioned. Given dropout's effect at lr=6e-4 was large (-0.029 to -0.090, well
outside noise), I expect this single run to be similarly decisive one way or the
other.

## Implementation
`configs/autoencoder.yaml`: `lr: 7.5e-4 -> 7e-4`, `dropout_rate: 0.0 -> 0.05`.
Everything else unchanged: weight_decay=0.0, noise_std=0.0, patience=50,
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