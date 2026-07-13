---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2      # the AE architecture trained (e.g. AE3dDilatedAttention). Descriptive, NOT the id.
summary: "Replicate 1/2 of the champion, no HP change: characterizing run-to-run variance (GPU nondeterminism, seed fixed) before trusting the +0.0079 margin over baseline"         # one-line description of the change (becomes the CSV modification_description)
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
All 5 open hyperparameters are now individually bracketed at the champion's exact
values, each direction of change tested and found neutral-or-worse. But several
of the deltas recorded along the way (weight_decay: -0.0017, noise_std's own gain:
+0.0079) are comparable to or smaller than the ~0.03-0.04 run-to-run noise floor
dim=60 measured under identical config/seed (GPU non-determinism at batch_size=1).
Before writing up this campaign's HP recommendation, I want at least a rough
variance estimate at dim=8 specifically — it may differ from dim=60's. This trial
(and one more after it) reruns the exact champion config with zero HP change, to
get n=3 total (including the original 02343abb run) and see whether 0.8056 sits
comfortably above the baseline or within noise of it.

## Implementation
No change to `configs/autoencoder.yaml` — identical to the current CHAMPION
(02343abb): lr=6e-4, weight_decay=0.0, dropout_rate=0.0, noise_std=0.0,
patience=50, latent_dimensions=8, seed=0 (unchanged, GPU non-determinism is the
only source of run-to-run variation here).

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