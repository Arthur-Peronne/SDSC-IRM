---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2      # the AE architecture trained (e.g. AE3dDilatedAttention). Descriptive, NOT the id.
summary: "Light regularization (weight_decay=1e-5, dropout_rate=0.1) on default HPs, first trial of the classification-accuracy campaign"         # one-line description of the change (becomes the CSV modification_description)
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
This is the campaign's first trial (no champion to branch from), so it doubles as the
reference point AND a deliberate first candidate rather than a bare no-op. The judge is
now downstream classification accuracy (logistic regression on 20-d AE latents, 100
train / 20 val patients), not AE reconstruction R2. With only 100 training patients and
a fairly deep separable-residual 3D encoder, the AE can plausibly overfit the training
patients' individual anatomy into the latent code — which helps reconstruction R2 but
can hurt the *generalization* of the latent geometry that the downstream classifier
relies on for unseen validation patients. I add light L2 (weight_decay=1e-5) to keep
encoder weights from growing large/idiosyncratic, and light dropout (dropout_rate=0.1)
to discourage any single latent dimension from being a memorized shortcut, forcing more
redundant/robust structure across the 20 dimensions. I predict this increases
classification_accuracy_val relative to the fully unregularized default, at the cost of
a small amount of reconstruction R2 (logged via also_log_ae, non-decisional). Keeping
lr=5e-4 and patience=20 unchanged: lr=5e-4 was the stable/best setting in the prior
(smoke-test) campaign at this architecture, and patience=20 already allows a real
early-stopping search within the 200-epoch cap, so I'm not conflating this trial's
regularization test with an unrelated optimizer-schedule change. noise_std stays 0.0 -
denoising is a distinct mechanism reserved for a later trial if this one doesn't clearly
help, to keep this first change interpretable.

## Implementation
In `configs/autoencoder.yaml` (the only mutable file), from the campaign defaults:
- `weight_decay: 0.0 -> 1e-5` (light L2 on all trainable weights)
- `dropout_rate: 0.0 -> 0.1` (light dropout, presumably applied in the encoder/decoder
  blocks per the existing `AE3dAsymResSeparableV2` implementation - architecture code
  itself is frozen/untouched)
- Unchanged: `lr=5e-4`, `noise_std=0.0`, `patience=20`, `latent_dimensions=20`,
  `model_name=AE3dAsymResSeparableV2`. No skip connections introduced (none touched -
  architecture file is not in `mutable` this campaign).

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