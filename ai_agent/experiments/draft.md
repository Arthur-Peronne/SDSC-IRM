---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2SELate
summary: "HP tuning on champion architecture (no code change): lr 1e-4 -> 2e-4, completing the lr sweep after 5e-5 widened seed variance instead of reducing it"
parent: 761cab78

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
Completing the `lr` sweep started by `d8bc06e5` (`lr=5e-5`, FAILURE 0.675): that trial's mechanistic
prediction about smoother optimization held (tightest best-epoch spread of the campaign, 39-49), but
classification-accuracy variance across seeds got WORSE, not better (0.750/0.600/0.675, a 0.15 spread,
the widest this campaign) — decoupling "smooth optimization" from "low classification variance."
Only the lower-`lr` direction has been tested; testing `lr=2e-4` (double the original) checks whether
this is a general effect of moving `lr` away from `1e-4` in either direction, or specific to going
lower. If accuracy variance is similarly high or accuracy drops here too, that's further evidence `lr`
in this range isn't a lever for the noise floor `program.md` documents, and the champion's `lr=1e-4`
is more likely a local optimum already worth leaving alone. If instead this improves or tightens
variance, it suggests the champion was previously under-trained per-step, not over-tuned.

## Implementation
No new class — reuses `AutoEncoder3D_AsymResSeparableV2_SELate` (model_name
`AE3dAsymResSeparableV2SELate`), same code as parent `761cab78`. Only `configs/autoencoder.yaml`
changes: `lr: 1e-4 -> 2e-4` (model_name/models_list reset to `AE3dAsymResSeparableV2SELate` after the
previous trial's auto-revert). All other hyperparameters (`weight_decay=0`, `dropout_rate=0`,
`noise_std=0`, `patience=20`) and the architecture itself are byte-for-byte the champion's.

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