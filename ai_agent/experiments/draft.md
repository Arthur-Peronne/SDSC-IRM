---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2      # the AE architecture trained (e.g. AE3dDilatedAttention). Descriptive, NOT the id.
summary: "dropout_rate 0.0 -> 0.01 on top of the champion: confirming dropout is harmful at a smaller magnitude too, not just 0.05"         # one-line description of the change (becomes the CSV modification_description)
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
Trial 5 (dropout=0.05) was harmful (-0.0286), a real effect. At dim=60, dropout
was harmful at every magnitude tested (0.01 and 0.05), non-monotonically (0.01
was actually *worse* than 0.05 in one context there). I test dropout=0.01 at
dim=8 to check the same non-monotonicity/robustness: does a smaller dropout still
hurt, confirming the "any explicit stochastic regularization is redundant on this
bottleneck" story, or does the direction flip at a small enough magnitude
(suggesting a narrow useful window I'd have missed)? Given dim=60's own
non-monotonic result, I don't have a strong prior on whether 0.01 is better or
worse than 0.05 — only that I expect it to remain in "harmful or at best neutral"
territory, not to reveal a clearly beneficial regime.

## Implementation
`configs/autoencoder.yaml`: `dropout_rate: 0.0 -> 0.01`. Built on top of the
current CHAMPION (02343abb): lr=6e-4, weight_decay=0.0, noise_std=0.0,
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