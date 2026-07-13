---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2
summary: "Explore an unexplored corner of the dropout axis — 0.01, 5x smaller than the 0.05 that consistently failed badly, to test whether a much gentler dose behaves like weight_decay (near-neutral) rather than like dropout=0.05 (harmful)"
parent: bed745a0

# ---- driver-written (leave null; the driver overwrites at lock/result) ----
id: null
status: draft
verdict: null
created_at: null
metric:
  primary: {name: avg_validation_R2_mean, value: null, direction: maximize}
---

# Trial <id> — <model_name> — <verdict>

<!-- ===== written BEFORE the run (agent) ===== -->

## Hypothesis
dropout_rate=0.05 failed badly and consistently (twice, under two different
schedules), always by cutting train R2 far more than it helped val R2. But only 0.05
was ever tested — a much smaller dose (0.01) has a qualitatively different effect:
at low rates, dropout approximates a mild noise-injection closer in spirit to
noise_std (which helped) than to a real capacity cut. I predict this is more likely
to be near-neutral (like weight_decay) than to reproduce 0.05's large failure,
though given the established ~0.03-0.04 noise floor, distinguishing "near-neutral"
from "small genuine win" will likely not be possible from a single run — the main
question this answers is whether dropout is harmful at ANY nonzero rate, or only
above some threshold, completing the dropout axis's characterization.

## Implementation
Single-field change in configs/autoencoder.yaml: dropout_rate 0.0 -> 0.01, on top of
the champion's lr=6e-4, noise_std=0.0001, patience=60. weight_decay=0 unchanged. No
architecture change.

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
