---
# Copy this file to draft.md, fill the AGENT-WRITTEN fields, then run:
#   python ai_agent/driver.py run
# The driver commits the input (that commit's short sha becomes `id`), renames
# draft.md -> <id>.md, trains, and fills the DRIVER-WRITTEN fields + ## Results.
# (These comments are not preserved in the final <id>.md — that's expected.)

# ---- agent-written (fill BEFORE running) ----
model_name: AE3dAsymResSeparableV2
summary: "Control replicate: exact champion config (lr=8e-4, dropout=0.05), no HP change, to measure run-to-run variance at this specific sharp optimum"
parent: 3e07b08d

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
Trials 274ff200/0d8678eb/3faa0f66 showed the champion's `dropout_rate=0.05` sits at a
sharp, narrow peak: every tested neighbor (0.03, 0.06, 0.08) degraded clearly and
consistently (shorter productive training, worse val_loss). Separately, trial
d78769c1 established a ~0.03 R2 run-to-run noise floor even for a nominally
unchanged config (seed=0 does not give bit-exact reproducibility here, likely due to
non-deterministic cuDNN kernels). This trial makes NO HP change — it is a literal
repeat of the champion's exact config — to directly measure how much
`avg_validation_R2_mean` varies at THIS specific operating point (not the old
baseline's, where the noise floor was originally measured). If the repeat lands
close to 0.828, the champion's edge is likely a real, robust effect of this specific
combination; if it lands far below (toward 0.80 or lower, comparable to the failed
neighbors), the sharp "peak" may be partly a lucky draw rather than a fully
reproducible advantage. This is deliberate, not idle: it directly informs how much
confidence to place in the campaign's best finding.

## Implementation
No change to `configs/autoencoder.yaml` relative to the champion (`lr=8e-4`,
`dropout_rate=0.05`, all other fields at baseline defaults). `seed=0` is also
unchanged (not touched, per program.md's restriction to the 5 opened HPs) — any
difference in outcome is due to non-deterministic training, not a deliberate input
change.

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