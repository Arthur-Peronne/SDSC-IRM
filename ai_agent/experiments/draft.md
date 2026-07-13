---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2      # the AE architecture trained (e.g. AE3dDilatedAttention). Descriptive, NOT the id.
summary: "Replicate of the BASELINE config (noise_std=0.0001), no HP change: fair n=2 mean-vs-mean comparison against the champion's n=3 mean given the confirmed noise floor"         # one-line description of the change (becomes the CSV modification_description)
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
The champion's 3-point spread (0.8056/0.7655/0.7991, mean 0.7901) now sits at or
below the single-run BASELINE value (0.797717) — the noise_std=0 "improvement"
looks like it may have been a favorable draw rather than a real effect. But
that's an n=3-vs-n=1 comparison, which is itself unfair. I replicate the BASELINE
config (noise_std=0.0001, everything else identical) once to get a same-footing
n=2 mean for the baseline before concluding the noise_std axis is genuinely flat
rather than mildly beneficial at dim=8.

## Implementation
`configs/autoencoder.yaml`: `noise_std: 0.0 -> 0.0001` (reverting to the
BASELINE's value), everything else unchanged: lr=6e-4, weight_decay=0.0,
dropout_rate=0.0, patience=50, latent_dimensions=8, seed=0.

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