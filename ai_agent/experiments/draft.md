---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2SELateDeepFC      # the AE architecture trained (e.g. AE3dDilatedAttention). Descriptive, NOT the id.
summary: "Nonlinear FC bottleneck: fc_enc/fc_dec through a 128-d ReLU hidden layer instead of one Linear"         # one-line description of the change (becomes the CSV modification_description)
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
Third distinct architectural family this campaign (after SE placement and pooling mechanism, both
now closed directions). Champion `AE3dAsymResSeparableV2SELate`'s `fc_enc`/`fc_dec` are each a
single `nn.Linear` (2048<->20): the 20-d latent is forced to be a purely linear function of the
flattened bottleneck features, and reconstruction a purely linear expansion back. I insert a 128-d
ReLU hidden layer on both sides (2048->128->20 and 20->128->2048). Mechanism: a nonlinear
compression can express feature interactions a linear map cannot (e.g. "high in channel A AND low
in channel B" style combinations) — if any of the group-discriminative structure in the 2048-d
bottleneck features is not linearly separable, only a nonlinear projection can preserve it into the
fixed 20-d budget. This is a genuinely different lever from SE (which reshapes what's IN the
2048-d vector) or pooling (which reshapes what survives spatially) — it changes how that vector is
compressed to 20 dims. Same encoder/decoder conv path, SE placement (se3, se4), and hyperparameters
(`lr=1e-4`, `weight_decay=0`, `dropout_rate=0`, `noise_std=0`, `patience=20`) as the champion.

## Implementation
New class `AutoEncoder3D_AsymResSeparableV2_SELateDeepFC` in `src/models/ae_models.py` (model_name
`AE3dAsymResSeparableV2SELateDeepFC`), copied from `AutoEncoder3D_AsymResSeparableV2_SELate`
(current champion) with `fc_enc`/`fc_dec` changed from a single `nn.Linear` each to
`nn.Sequential(Linear(in, 128), ReLU, Linear(128, out))`. Dropout is still applied at the same points
as the champion (after flatten before `fc_enc`, after `fc_dec` before reshaping) — inert either way
since `dropout_rate=0` this campaign. Everything else (conv encoder/decoder, SE) unchanged. No
encoder-to-decoder path — respects the no-skip-connections rule in `program.md`. Verified the new
class builds, forward-passes, and produces a (1,20) latent before launching the trial (parameter
count grows from ~1.13M to ~1.58M due to the two hidden FC layers, still a small model for this
task).

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