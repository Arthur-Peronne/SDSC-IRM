---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2      # the AE architecture trained (e.g. AE3dDilatedAttention). Descriptive, NOT the id.
summary: "patience 50 -> 70 on top of the champion: does a slower LR-decay cadence keep helping past 50, or plateau/reverse like dim=60 saw?"         # one-line description of the change (becomes the CSV modification_description)
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
Trial 1181ebc7 (patience=30) showed a real cost (-0.0193): shorter patience means
a faster patience_scheduler cadence (patience//5), which decays lr sooner and
stops the run before it can refine to as good a plateau. By the same mechanism,
a longer patience=70 (patience_scheduler=14, vs 10 at patience=50) should allow
more/later lr-decay steps and could reach an even better plateau — mirroring the
shorter-patience cost but in reverse. At dim=60 though, patience=60 (scheduler=12)
was not clearly better than 50 (scheduler=10) — comparable results, higher
variance — so I expect a small gain at best, possibly flat, not a repeat of the
large gains seen on the noise_std/dropout axes.

## Implementation
`configs/autoencoder.yaml`: `patience: 50 -> 70` (patience_scheduler auto =
patience // 5 = 14, vs 10 before). Built on top of the current CHAMPION
(02343abb): lr=6e-4, weight_decay=0.0, dropout_rate=0.0, noise_std=0.0,
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