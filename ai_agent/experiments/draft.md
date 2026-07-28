---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2SELate      # the AE architecture trained (e.g. AE3dDilatedAttention). Descriptive, NOT the id.
summary: "HP check on champion arch: mild regularization (weight_decay=1e-5, dropout_rate=0.1)"         # one-line description of the change (becomes the CSV modification_description)
parent: 761cab78         # lineage: `id` of the trial whose CODE this one branched FROM. null for roots.

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
Deliberate HP check, not a random sweep, motivated directly by the last 2 trials (00c71333/CBAM and
especially c0a7bad4/DeepFC): both added capacity with zero regularization (`weight_decay=0`,
`dropout_rate=0`, held fixed across every architecture trial so far to isolate the architecture
variable) and both underperformed, with DeepFC also showing slower/more variable convergence
consistent with mild overfitting. This suggests the champion architecture itself may currently be
regularization-starved rather than needing more architectural capacity. This trial keeps the
champion `AE3dAsymResSeparableV2SELate` architecture unchanged and only re-introduces mild
regularization: `weight_decay=1e-5` (the original pre-campaign default) and `dropout_rate=0.1` (a
mild value, well below the original 0.3 default). Prediction: if the champion was leaving accuracy
on the table due to overfitting on the 100-patient training set, mild regularization should improve
or at least not hurt `classification_accuracy_val`, and should reduce seed-to-seed variance.
`lr=1e-4`, `patience=20`, `noise_std=0` unchanged.

## Implementation
No code change to `src/models/ae_models.py` — `AE3dAsymResSeparableV2SELate` is unchanged, this is
purely a `configs/autoencoder.yaml` hyperparameter change: `weight_decay` 0.0 -> 1e-5, `dropout_rate`
0.0 -> 0.1. This is a deliberate return to HP tuning after 3 consecutive architecture trials
(CBAM, strided pooling, deep FC) failed, per `program.md`'s guidance to interleave architecture and
HP work rather than defer HP entirely to the end — this check is directly motivated by a pattern
observed across those trials, not exploratory random search.

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