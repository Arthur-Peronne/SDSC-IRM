---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2      # the AE architecture trained (e.g. AE3dDilatedAttention). Descriptive, NOT the id.
summary: "Campaign baseline at latent_dim=60, full budget (n_epochs=200) — HP reset to neutral/near-zero values (lr=1e-5, weight_decay=0, dropout_rate=0, noise_std=0, patience=30) rather than carried from the dim=240 champion, since a 4x smaller bottleneck likely needs a different regularization/lr operating point"
parent: null          # lineage: `id` of the trial whose CODE this one branched FROM. null for roots.

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
This is the root trial of a new campaign at latent_dimensions=60 (previous campaign
optimized the same architecture at latent_dimensions=240). Rather than carry over the
dim=240 champion's HPs (lr=8e-4, dropout_rate=0.05), I start from neutral/near-zero
values (lr=1e-5, weight_decay=0, dropout_rate=0, noise_std=0, patience=30) because a
4x smaller latent bottleneck changes the optimization landscape: less capacity to
overfit means the regularization (dropout, weight_decay, noise) that helped at dim=240
may not be needed or may even hurt convergence at dim=60, and the learning rate that
was optimal for a wider/deeper effective path at dim=240 is not assumed to transfer.
This baseline establishes the reference avg_validation_R2_mean against which
subsequent HP trials at this latent_dim are judged.

## Implementation
No code change. configs/autoencoder.yaml already has latent_dimensions=60 and the 5
opened hyperparameters at: lr=1e-5, weight_decay=0.0, dropout_rate=0.0, noise_std=0.0,
patience=30. n_epochs=200 (early stopping via patience, n_val=20>0, compute_metrics=true).
This trial commits that state as-is to establish the BASELINE.

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
