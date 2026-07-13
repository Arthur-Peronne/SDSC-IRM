---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2      # the AE architecture trained (e.g. AE3dDilatedAttention). Descriptive, NOT the id.
summary: "Baseline replicate 3/6 (noise_std=0.0001): building a matched n=6 sample against the champion's n=6 for a fair mean-vs-mean comparison"         # one-line description of the change (becomes the CSV modification_description)
parent: 0d2e0fa2          # lineage: `id` of the trial whose CODE this one branched FROM. null for roots.

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
The champion (noise_std=0) now has a solid n=6 estimate (mean 0.7817, std≈0.0238).
The baseline (noise_std=0.0001) only has n=2 (mean 0.7710) — too few to compare
fairly. Rather than declare noise_std's effect settled on an unequal footing, I am
building the baseline up to a matched n=6 (this is replicate 3 of 4 needed:
trials 19-22) so the two means can be compared on equal terms. No new hypothesis
on the HP itself — this is about sample-size fairness for the campaign's final
conclusion.

## Implementation
`configs/autoencoder.yaml`: `noise_std: 0.0 -> 0.0001` (baseline's value),
everything else unchanged: lr=6e-4, weight_decay=0.0, dropout_rate=0.0,
patience=50, latent_dimensions=8, seed=0.

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