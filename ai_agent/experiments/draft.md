---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2
summary: Increase dropout_rate 0.3->0.5 on champion (lr=8e-4, weight_decay=1e-5), testing how far dropout can be pushed before the R2/accuracy tradeoff reverses
parent: a2a3d9d1

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
Trial 3 showed that `dropout=0.3` + `lr=5e-4` = 0.6083 beats `dropout=0.1` + `lr=5e-4` (baseline: 0.5917). The current champion confirms this at `lr=8e-4`: `dropout=0.3` (0.6250) > `dropout=0.1` (0.6083, trial 12). The question is whether the trend continues at `dropout=0.5`. Mechanistically, higher dropout on the 2048-d bottleneck forces even more redundancy in the latent representation, which should help the classifier generalize better. However, there's a tradeoff: too much dropout degrades reconstruction quality (R²), and if the latent codes become too noisy, the classifier can't use them effectively. The campaign summary specifically asked: "how far can dropout_rate be pushed before the R2/accuracy tradeoff reverses?" I predict `dropout=0.5` still improves accuracy over 0.3, but R² will drop noticeably.

## Implementation
In `configs/autoencoder.yaml` (only mutable file), branching from champion a2a3d9d1:
- `dropout_rate: 0.3 -> 0.5` (bottleneck dropout only, aggressive regularization)
- Unchanged: `lr=8e-4`, `weight_decay=1e-5`, `noise_std=0.0`, `patience=20`, `model_name=AE3dAsymResSeparableV2`, `latent_dimensions=20`

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