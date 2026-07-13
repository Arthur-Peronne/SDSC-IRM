---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2
summary: "Control replicate #1 — exact champion config (lr=6e-4, noise_std=0.0001, patience=60), no HP change, to measure run-to-run variance under batch_size=1 despite the fixed seed"
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
The systematic 5-axis sweep and its main interactions are now exhaustively mapped
(18 trials); every remaining champion-to-champion delta reported so far (as small as
+0.0027 for noise_std, or -0.0040 for weight_decay) has no error bar. seed=0 is fixed,
but the dim=240 campaign's own control replicates showed real run-to-run variance
despite a fixed seed (likely non-deterministic CUDA ops at batch_size=1), and this
campaign's own trials have shown similar-magnitude noise in single-epoch val spikes.
I run the exact champion config unchanged to get the first of 2-3 replicate points,
establishing how much of the campaign's small late-stage deltas (the last several
CHAMPION/FAILURE margins have all been under 0.01) could be noise rather than signal.

## Implementation
No change to configs/autoencoder.yaml — exact champion config (lr=6e-4,
weight_decay=0.0, dropout_rate=0.0, noise_std=0.0001, patience=60). No architecture
change.

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
