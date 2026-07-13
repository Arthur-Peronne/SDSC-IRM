---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2      # the AE architecture trained (e.g. AE3dDilatedAttention). Descriptive, NOT the id.
summary: "lr 6e-4 -> 8e-4 on top of the zero-regularization champion: does the lr ceiling shift now that noise_std=0?"         # one-line description of the change (becomes the CSV modification_description)
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
At dim=60, lr=8e-4 was a clean FAILURE (-0.008 to -0.019 depending on context),
but that was always tested with noise_std=0.0001 present, i.e. the optimizer was
fighting input noise on top of the lr step size. Now that three trials (2, 4, 5)
have shown dim=8's zero-regularization champion has no explicit regularizer left
(weight_decay=0, dropout=0, noise_std=0), the "why 8e-4 failed" mechanism at
dim=60 (excess step size interacting badly with noise-induced gradient variance)
may not apply the same way here — there is no noise left to interact with. I test
lr=8e-4 directly on top of the current champion to see whether the ceiling is
still 6e-4 in this cleaner (zero-noise) optimization landscape, or whether it can
now tolerate a larger step. This is a genuine open question, not a confirmatory
check like trials 2/5 — I do not have a strong prior on the sign here.

## Implementation
`configs/autoencoder.yaml`: `lr: 6e-4 -> 8e-4`. Built on top of the current
CHAMPION (02343abb): weight_decay=0.0, dropout_rate=0.0, noise_std=0.0,
patience=50, latent_dimensions=8.

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