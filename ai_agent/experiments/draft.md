---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2      # the AE architecture trained (e.g. AE3dDilatedAttention). Descriptive, NOT the id.
summary: "weight_decay 0 -> 1e-6, all else unchanged (dim=60 report flagged this exact combo as never tested)"         # one-line description of the change (becomes the CSV modification_description)
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
At dim=60, the report's §4.2 found a PRELIMINARY (n=1, unreplicated) hint that
weight_decay dampens the instability caused by input noise (tested only alongside
noise_std=0.0002, which fails badly alone). weight_decay alone was mildly neutral-
to-negative (§4.1). The combination `noise_std=0.0001 + weight_decay=1e-6` — the
champion's own noise level plus a small weight_decay — was explicitly flagged in
§8 as never tested and "a direct, low-cost first check for a follow-up campaign".
I add `weight_decay=1e-6` to the dim=8 baseline (0d2e0fa2) unchanged otherwise. I
predict a small effect (within/near the noise band), since dim=60's weight_decay-
alone effect was itself small and slightly negative — this is a deliberate,
report-directed check, not an expectation of a large gain.

## Implementation
`configs/autoencoder.yaml`: `weight_decay: 0.0 -> 1e-6`. Everything else identical
to the BASELINE (0d2e0fa2): lr=6e-4, dropout_rate=0.0, noise_std=0.0001,
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