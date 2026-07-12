---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2
summary: "Map the noise_std cliff boundary — test 0.0002 (between the winning 0.0001 and the badly-failing 0.0003) on top of the champion's lr=6e-4"
parent: 319dacea

# ---- driver-written (leave null; the driver overwrites at lock/result) ----
id: null
status: draft
verdict: null
created_at: null
metric:
  primary: {name: avg_validation_R2_mean, value: null, direction: maximize}
---

# Trial <id> — <model_name> — <verdict>

<!-- ===== written BEFORE the run (agent) ===== -->

## Hypothesis
noise_std=0.0001 won (+0.0027) but 0.0003 failed badly (-0.0523) — a sharp,
non-monotonic cliff rather than a smooth diminishing-returns curve like lr's. I test
the midpoint (0.0002) to locate the boundary: if it is close to 319dacea's result,
the cliff is nearer 0.0003 and there may be a slightly better point than 0.0001
between them; if it is already noticeably worse (closer to 5722444c's -0.0523), the
cliff is sharp and immediate right above 0.0001, meaning 0.0001 is likely at or very
near the true local optimum for this mechanism and not worth fine-tuning further.

## Implementation
Single-field change in configs/autoencoder.yaml: noise_std 0.0001 -> 0.0002, on top
of the champion's lr=6e-4. weight_decay=0, dropout_rate=0, patience=30 unchanged. No
architecture change.

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
