---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2      # the AE architecture trained (e.g. AE3dDilatedAttention). Descriptive, NOT the id.
summary: "patience 50 -> 30 on top of the champion: does dim=8's faster convergence (best epochs 22-76 so far) tolerate a shorter patience without losing the optimum?"         # one-line description of the change (becomes the CSV modification_description)
parent: 02343abb          # lineage: `id` of the trial whose CODE this one branched FROM. null for roots.

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
Patience is the last of the 5 open hyperparameters, untouched so far at dim=8. At
dim=60, the report found patience mainly affected run-to-run reproducibility
(variance), not the performance ceiling — best result (0.8148) was still recorded
at patience=50/60. At dim=8, best epochs observed so far cluster early (22-76
across trials, all well under patience=50's 50-epoch stall window), suggesting the
model converges faster at this smaller capacity and may not need as long a
patience to find its optimum. I predict a mostly neutral effect on the metric
itself (consistent with dim=60), possibly with a small speed/stability trade-off,
rather than a large gain or loss — this is a closing/completeness check for the
axis, not a strong lead.

## Implementation
`configs/autoencoder.yaml`: `patience: 50 -> 30` (patience_scheduler auto =
patience // 5 = 6, vs 10 before). Built on top of the current CHAMPION
(02343abb): lr=6e-4, weight_decay=0.0, dropout_rate=0.0, noise_std=0.0,
latent_dimensions=8.

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