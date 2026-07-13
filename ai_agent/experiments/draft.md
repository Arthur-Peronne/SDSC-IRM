---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2
summary: "Fourth control replicate of the champion (patience=60), balancing the sample size against patience=45's 3 replicates for a fairer variance comparison"
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
patience=60 has 4 known draws (0.8148, 0.8050, 0.7396, 0.7783, spread 0.075);
patience=45 has 3 draws (0.8089, 0.8060, 0.7913, spread 0.018), suggesting materially
lower variance for the shorter schedule. Before treating this as a settled
conclusion, I want one more patience=60 draw to check whether its apparent wider
spread holds up, or whether the low points (0.7396 especially) were themselves
unusual and a 5th draw clusters closer to the top three (0.8148, 0.8050, 0.7783).

## Implementation
No change to configs/autoencoder.yaml — exact champion config (lr=6e-4,
weight_decay=0.0, dropout_rate=0.0, noise_std=0.0001, patience=60). No architecture
change. Fifth replicate of bed745a0's config.

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
