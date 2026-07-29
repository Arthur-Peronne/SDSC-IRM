---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2SELateEnc3Only
summary: "HP tuning on champion architecture (no code change): patience 20 -> 35, testing whether the recurring weak-seed pattern improves with more room to escape an early plateau"
parent: ac5057cf

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
Only 4 trials remain (`lr` is closed both directions, two fusion attempts and one SE-refinement have
all failed on top of champion `ac5057cf`) — `patience` (currently 20) is the one hyperparameter never
touched this entire campaign. A recurring pattern across recent trials (`ac5057cf`, `b02b8293`,
`0afac601`, `582021d1`) is "two seeds land near 0.70-0.75, one seed lands notably weaker," and several
weak seeds show fast, shallow convergence (e.g. `582021d1`'s seed 0, best epoch 13) — consistent with
early stopping cutting off a seed before it has a chance to escape an early plateau via the LR-decay
schedule (which only kicks in after `patience` epochs of no improvement, and decays further every
`patience//5` epochs after that). Raising `patience` to 35 gives every seed more LR-decay steps before
stopping, which I predict either helps the weak seeds catch up (improving the aggregate) or is
neutral if `ac5057cf`'s champion result already reflects each seed's true achievable optimum within
its own dynamics. No code change — pure hyperparameter test.

## Implementation
No new class — reuses `AutoEncoder3D_AsymResSeparableV2_SELateEnc3Only` (model_name
`AE3dAsymResSeparableV2SELateEnc3Only`), the exact champion `ac5057cf` code. Only
`configs/autoencoder.yaml` changes: `patience: 20 -> 35`. All other hyperparameters (`lr=1e-4`,
`weight_decay=0`, `dropout_rate=0`, `noise_std=0`) and the architecture itself are byte-for-byte the
champion's.

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