---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2      # the AE architecture trained (e.g. AE3dDilatedAttention). Descriptive, NOT the id.
summary: "weight_decay 0.0 -> 1e-5 on top of the champion: last untested single-axis point (only 1e-6 tested so far), closing the axis before pivoting to final replication"         # one-line description of the change (becomes the CSV modification_description)
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
weight_decay=1e-6 was neutral (-0.0017, well inside noise). At dim=60, 1e-5 was
slightly *more* negative than 1e-6 (-0.008 vs -0.004 there), a small but
consistent direction (never positive). I test 1e-5 at dim=8 mainly for
completeness — expecting a similarly small, probably noise-bound effect, likely
in the same direction (neutral-to-mildly-negative) rather than uncovering a
missed beneficial regime. This closes the weight_decay axis with two tested
points instead of one, mirroring how thoroughly lr/dropout/patience were each
checked.

## Implementation
`configs/autoencoder.yaml`: `weight_decay: 0.0 -> 1e-5`. Built on top of the
current CHAMPION (02343abb): lr=6e-4, dropout_rate=0.0, noise_std=0.0,
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