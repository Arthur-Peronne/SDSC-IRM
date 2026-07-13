---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2
summary: "Third replicate of the recorded champion config (patience=50), which currently has only 2 points (0.8148, 0.7796) — the least-characterized of the near-tied top candidates"
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
The ledger's recorded champion (bc589070, patience=50, patience_scheduler=10) has
only 2 replicate points (0.8148, 0.7796, spread 0.035), fewer than patience=45 (n=4)
or patience=49 (n=3). Before finalizing the campaign's comparison across nearby
patience values, I want a 3rd point for the actual recorded champion to put it on
comparable footing. Given patience_scheduler=10 groups with 60's wide-variance
pattern rather than 45/49's tighter one (per the emerging but tempered
scheduler-cadence pattern), I expect this draw to plausibly fall anywhere in the
~0.74-0.82 range already seen for scheduler>=10 configs.

## Implementation
No change to configs/autoencoder.yaml — exact champion config (lr=6e-4,
weight_decay=0.0, dropout_rate=0.0, noise_std=0.0001, patience=50). No architecture
change. Third replicate of bc589070's config.

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
