---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2      # the AE architecture trained (e.g. AE3dDilatedAttention). Descriptive, NOT the id.
summary: "lr 6e-4 -> 3e-4, all else unchanged (checking whether the dim=60 lr-optimum shifts at dim=8)"         # one-line description of the change (becomes the CSV modification_description)
parent: 0d2e0fa2          # lineage: `id` of the trial whose CODE this one branched FROM. null for roots.

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
At dim=60, lr showed a clean monotonic-then-cliff curve: 1e-5 -> 6e-4 improving
(diminishing returns), then a clean failure at 7e-4/8e-4 (§1 of the dim=60 report).
That peak was found on a ~60-dim bottleneck; dim=8 is a much smaller model with
far fewer effective degrees of freedom near the bottleneck, so the loss landscape
there could be sharper/more sensitive to step size. I test one point below the
transferred baseline (3e-4, which was itself CHAMPION at dim=60, R²=0.7874, just
short of the 6e-4 peak) to see whether the dim=8 optimum sits lower than 6e-4. If
3e-4 beats the 0.7977 baseline, the lr-peak shifts down with capacity and I'll
refine further below it; if it's flat or worse, 6e-4 stands confirmed at dim=8 too
and I'll stop probing this axis and move to noise_std/dropout instead.

## Implementation
`configs/autoencoder.yaml`: `lr: 6e-4 -> 3e-4`. Everything else identical to the
BASELINE (0d2e0fa2): weight_decay=0.0, dropout_rate=0.0, noise_std=0.0001,
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