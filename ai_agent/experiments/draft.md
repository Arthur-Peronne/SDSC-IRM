---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2      # the AE architecture trained (e.g. AE3dDilatedAttention). Descriptive, NOT the id.
summary: "lr 6e-4 -> 7e-4 on top of the champion: refining the lr axis between the confirmed-good 6e-4 and the catastrophic 8e-4, single-run screening pass"         # one-line description of the change (becomes the CSV modification_description)
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
lr is the axis with by far the largest, clearest effect sizes found so far
(3e-4: -0.025; 8e-4: -0.103, both well outside the noise floor). Rather than
chasing marginal, noise-bound differences on weight_decay/noise_std further, I am
pivoting the remaining budget to pin down the lr curve more precisely: is there a
clean cliff somewhere between 6e-4 (good) and 8e-4 (catastrophic), or a gradual
decline? Testing 7e-4 first as a single-run screen — if the result is far from
baseline (either direction), that's already informative at n=1 given this axis's
demonstrated large effect sizes; only if it lands ambiguously close to the
champion would I spend replicates confirming it.

## Implementation
`configs/autoencoder.yaml`: `lr: 6e-4 -> 7e-4`. Built on top of the current
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