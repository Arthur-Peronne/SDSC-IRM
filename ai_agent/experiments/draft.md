---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2
summary: "First genuine regularizer COMBINATION test — weight_decay=1e-6 AND dropout_rate=0.01 stacked together (both individually mild/near-neutral alone), never tried simultaneously across the whole campaign"
parent: bc589070

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
Every regularizer test this campaign changed exactly ONE of weight_decay,
dropout_rate, noise_std at a time relative to the champion (which always keeps
noise_std=0.0001 fixed as background) — the pairwise interactions between
weight_decay and dropout_rate were never tested. Individually: weight_decay=1e-6 is
near-neutral (-0.004 to -0.008 across contexts), dropout_rate=0.01 is mildly-to-
moderately harmful (-0.054). I predict stacking them compounds toward dropout's
direction (since dropout's capacity cost dominates weight_decay's much smaller
penalty) — i.e. another FAILURE of similar or slightly worse magnitude than
dropout=0.01 alone — but a result close to weight_decay-alone's mild cost, or a
non-additive interaction (better or much worse than either alone), would be
genuinely new information about how these two capacity-affecting regularizers
combine, which single-axis testing cannot reveal.

## Implementation
Two-field change in configs/autoencoder.yaml: weight_decay 0.0 -> 1e-6 AND
dropout_rate 0.0 -> 0.01, simultaneously, on top of the champion's lr=6e-4,
noise_std=0.0001, patience=50 (patience left untouched per updated campaign
direction). No architecture change.

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
