---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2      # the AE architecture trained (e.g. AE3dDilatedAttention). Descriptive, NOT the id.
summary: "patience 50 -> 45 (patience_scheduler=9) on top of the champion: mirroring dim=60's own hint that scheduler=9 may be more reproducible at a comparable ceiling"         # one-line description of the change (becomes the CSV modification_description)
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
Patience=30 (scheduler=6) and 70 (scheduler=14) both failed clearly, bracketing
50 (scheduler=10) as a local optimum — but the dim=60 report flagged a specific,
narrower hint: patience 45/49 (scheduler=9) showed *lower variance* than 50/60
(scheduler=10-12) at a comparable performance ceiling there. That's a much finer
distinction than the 30/70 brackets I've tested — scheduler=9 vs 10 is one LR-
halving step apart, not several. I test patience=45 (scheduler=9) to see if this
narrow reproducibility difference also shows up at dim=8, since if it does, it
would be a genuinely useful practical recommendation (same ceiling, less
variance) even though it wouldn't change the primary metric's point estimate
much.

## Implementation
`configs/autoencoder.yaml`: `patience: 50 -> 45` (patience_scheduler auto =
patience // 5 = 9, vs 10 before). Built on top of the current CHAMPION
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