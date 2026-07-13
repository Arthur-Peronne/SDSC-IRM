---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2      # the AE architecture trained (e.g. AE3dDilatedAttention). Descriptive, NOT the id.
summary: "lr 7e-4 -> 7.5e-4: single-run screen to see if the newly-discovered better basin extends further toward the known 8e-4 cliff"         # one-line description of the change (becomes the CSV modification_description)
parent: 71508734          # lineage: `id` of the trial whose CODE this one branched FROM. null for roots.

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
lr=7e-4's replicated mean (0.7991, std≈0.0091) is well above lr=6e-4's
(0.7817). The cliff to catastrophic failure is at 8e-4 (mean -0.103 single run).
7.5e-4 sits between the confirmed-good 7e-4 and the confirmed-catastrophic 8e-4.
Testing whether the improved basin extends toward 7.5e-4 (continuing the
apparent trend: higher lr up to some point = better, within this fast-converging
regime) or whether 7e-4 is already close to a narrow local peak. Single-run
screen first, given the axis's history of large, informative single-run deltas.

## Implementation
`configs/autoencoder.yaml`: `lr: 7e-4 -> 7.5e-4`. Built on top of the current
CHAMPION (71508734): weight_decay=0.0, dropout_rate=0.0, noise_std=0.0,
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