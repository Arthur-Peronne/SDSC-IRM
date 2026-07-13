---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2      # the AE architecture trained (e.g. AE3dDilatedAttention). Descriptive, NOT the id.
summary: "lr=7.5e-4 replicate 3/3: confirming whether the huge n=2 spread (0.8114/0.7143) reflects real instability or a one-off bad draw"         # one-line description of the change (becomes the CSV modification_description)
parent: e5a91532          # lineage: `id` of the trial whose CODE this one branched FROM. null for roots.

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
lr=7.5e-4 n=2 shows a huge spread (0.8114/0.7143, std≈0.069) — much wider than
lr=7e-4's tight n=6 (std=0.0091). Mechanistically plausible: 7.5e-4 sits closer
to the confirmed 8e-4 instability cliff, so it may occasionally tip into a bad
basin even though its ceiling is higher. One more replicate (n=3) to see whether
this run lands near the good point (0.81ish) or the bad one (0.71ish) — informs
whether 7.5e-4's occasional instability is roughly as frequent as the n=2 sample
suggests (50/50) or rarer.

## Implementation
No change to `configs/autoencoder.yaml`: lr=7.5e-4, weight_decay=0.0,
dropout_rate=0.0, noise_std=0.0, patience=50, latent_dimensions=8, seed=0.

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