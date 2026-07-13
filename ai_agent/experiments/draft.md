---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2      # the AE architecture trained (e.g. AE3dDilatedAttention). Descriptive, NOT the id.
summary: "Champion replicate 3/N, no HP change: all 5 axes now closed, pivoting remaining budget to building a trustworthy final mean/CI for the recommended config"         # one-line description of the change (becomes the CSV modification_description)
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
All 5 open hyperparameters (lr, weight_decay, dropout_rate, noise_std, patience)
are now individually closed at, or statistically indistinguishable from, the
champion's values (trials 2-15). The champion currently has a 3-point replicate
spread (0.8056/0.7655/0.7991, mean 0.7901, std≈0.0215) that is itself close to
the baseline's 2-point spread (mean 0.7710). With ~24 trials of budget remaining
and no further single-axis leads worth chasing (marginal deltas are swamped by
noise), the most valuable use of the rest of the campaign is building a properly
powered estimate of the champion's true mean rather than more single-shot HP
probes. This is replicate 3 of an ongoing series (target ~6-8 total) to narrow
the standard error on the campaign's final reported number.

## Implementation
No change to `configs/autoencoder.yaml` — identical to the CHAMPION (02343abb)
and to replicates 925e559d/d84743ed: lr=6e-4, weight_decay=0.0, dropout_rate=0.0,
noise_std=0.0, patience=50, latent_dimensions=8, seed=0 unchanged.

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