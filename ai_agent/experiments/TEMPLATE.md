---
# Copy this file to draft.md, fill the fields below, then run the driver.
# The driver freezes `identity` at commit 1, computes `id`, and renames the file
# to <id>.md. `identity` is the hash preimage and is never edited after lock.
id: null                    # filled by driver at lock: sha256(canonical_json(identity))[:12]
parent: null                # lineage: trial this one branched FROM (code-wise). null for roots.
status: draft               # draft | running | completed | failed
identity:                   # FROZEN at commit 1 — never edit after lock
  parent: null
  commit: null              # code HEAD at lock (captures code + all configs + experiment.yaml)
  command: null             # the driver call (repeat_over lives in experiment.yaml)
model_name: null            # the AE architecture trained (e.g. AE3dDilatedAttention). Descriptive, NOT the id.
summary: null               # one-line description of the change (used as CSV modification_description)
metric:                     # only the PRIMARY aggregated metric is stored — the sole judge.
  primary: {name: avg_validation_R2_mean, value: null, direction: maximize}
created_at: null
---

# Trial <ID> — <ModelName> — <CHAMPION / CANDIDATE / FAILURE>

<!-- ===== written BEFORE the run (commit 1), by the agent ===== -->

## Hypothesis
<!-- What, why, how: "I modified X in Y because it will address Z via mechanism W."
     If this trial fuses two prior ideas, name both parent trials and the fusion here. -->

## Implementation
<!-- Key architectural / hyperparameter changes: what was added, where, and how it
     differs from the current champion. -->

<!-- ===== written AFTER the run (commit 2): numbers by the driver, prose by the agent ===== -->

## Results
- **R2_dim8:** VALUE | **R2_dim60:** VALUE | **R2_dim240:** VALUE
- **avg_validation_R2_mean:** VALUE
- **delta_vs_champion** (display only): VALUE
- **MLflow Run IDs:** RUNID_8 RUNID_60 RUNID_240
- **Best epochs:** EPOCH_8/N_EPOCHS | EPOCH_60/N_EPOCHS | EPOCH_240/N_EPOCHS

## Training Dynamics
<!-- Stability, convergence speed, notable observations (spikes, plateau, early stopping). -->

## Conclusion
<!-- Did the hypothesis hold? Mechanistic explanation of why it worked or failed. -->