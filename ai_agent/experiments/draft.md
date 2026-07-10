---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2      # the AE architecture trained (e.g. AE3dDilatedAttention). Descriptive, NOT the id.
summary: "lr 5e-4 -> 1.5e-3 (3x): test whether pushing lr further still helps, or the epoch4-5 wobble seen at 5e-4 turns into instability"
parent: e3b6e5ff      # lineage: `id` of the trial whose CODE this one branched FROM. null for roots.

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
The current champion (e3b6e5ff, lr=5e-4) converged in ~3 epochs then showed a mild
post-peak wobble in epochs 4-5 (val loss ticking back up, most visibly at dim=40:
0.0010 -> 0.0046 at epoch 5) — a classic early signature of the step size becoming too
large relative to the loss landscape's curvature near the optimum, even though it did
not cause outright divergence. This is the last trial of the campaign (`max_trials=3`),
so rather than a fresh direction I test the natural follow-up to trial 2's confirmed
lever: push `lr` further (5e-4 -> 1.5e-3, 3x) to see whether, under the same frozen
5-epoch budget, the faster-convergence benefit continues (if the wobble was mere noise
and convergence speed still dominates) or the increase now costs more than it gains
(if the wobble was an early warning of instability). Either outcome is informative for
future campaigns' choice of `lr`, even though this trial may not beat the champion.

## Implementation
`configs/autoencoder.yaml`: `lr: 5e-4` -> `lr: 1.5e-3`. No other field touched
(`weight_decay`, `dropout_rate`, `noise_std`, `patience` unchanged from the champion;
architecture, data, split, `n_epochs` all frozen/untouched).

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