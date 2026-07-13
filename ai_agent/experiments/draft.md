---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2      # the AE architecture trained (e.g. AE3dDilatedAttention). Descriptive, NOT the id.
summary: "dropout_rate 0.0 -> 0.05 on top of the new champion (noise_std=0): does dim=8 tolerate dropout any better than dim=60 did?"         # one-line description of the change (becomes the CSV modification_description)
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
At dim=60, dropout was harmful at every magnitude tested (0.01-0.05), consistently
costing more capacity than it recovered from reduced overfitting (§3 of the dim=60
report). Trial 02343abb just showed the same directional pattern for noise_std:
dim=8's bottleneck is already enough implicit regularization, and adding an
explicit one (denoising noise) hurt rather than helped. I predict dropout will
follow the same pattern, likely even more sharply — a dim=8 code has only 8 scalar
channels to carry all reconstructive signal, and randomly zeroing activations
upstream of that bottleneck should be proportionally more damaging than at dim=60
where there's more redundant capacity to spare. This is mainly a confirmatory
check (closing the axis) rather than an exploratory one — I do not expect a
different sign than dim=60, only possibly a larger-magnitude failure.

## Implementation
`configs/autoencoder.yaml`: `dropout_rate: 0.0 -> 0.05`. Built on top of the
current CHAMPION (02343abb): lr=6e-4, weight_decay=0.0, noise_std=0.0, patience=50,
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