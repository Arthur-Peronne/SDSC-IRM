---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2      # the AE architecture trained (e.g. AE3dDilatedAttention). Descriptive, NOT the id.
summary: "noise_std 0.0001 -> 0.0 (ablation): does the denoising mechanism still matter at dim=8?"         # one-line description of the change (becomes the CSV modification_description)
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
At dim=60, noise_std showed a sharp, non-monotonic cliff around a narrow sweet
spot: 0.0001 was CHAMPION (input-noise denoising acting as a mild regularizer),
but the ablation (noise_std=0.0, i.e. removing it) cost -0.012 there too — so the
effect wasn't just "any regularization helps," the exact value mattered. At dim=8
the bottleneck itself is already a much stronger implicit regularizer (7.5x fewer
latent dims), so denoising-as-regularization may be redundant here and add nothing
(or even remove useful signal the tiny bottleneck needs). I ablate noise_std to 0.0
to test whether the dim=8 baseline's R²=0.7977 depends on it at all, isolating
this axis before deciding whether to keep tuning it.

## Implementation
`configs/autoencoder.yaml`: `noise_std: 0.0001 -> 0.0`. Everything else identical
to the BASELINE (0d2e0fa2): lr=6e-4, weight_decay=0.0, dropout_rate=0.0,
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