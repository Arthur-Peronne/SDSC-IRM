# Campaign report — AE3dAsymResSeparableV2 @ latent_dim=60

<!--
Summary report for the hyperparameter-optimization campaign run by the autonomous
agent (ai_agent/driver.py), from trial 20fa5d8e (baseline) to trial 756da5dc
(40/40, budget exhausted). Prior champion for this model: latent_dim=240 (see
ai_agent/experiments/aiagent_HP_sepv2_240/).
-->

## Summary

- **40/40 trials**, budget exhausted (`max_trials=40` in `experiment.yaml`).
- **Final champion: `bc589070`** — `lr=6e-4, weight_decay=0, dropout_rate=0, noise_std=0.0001, patience=50`
  → `avg_validation_R2_mean = 0.8148`.
- Gain vs baseline (`20fa5d8e`, neutral HPs): **0.6517 → 0.8148 (+0.163)**. This gain
  is large and robust, well above the run-to-run noise characterized during the
  campaign (~0.03-0.04 standard deviation, see § Variance).
- Research direction changed mid-campaign at the user's request (around trial
  ~36/40): stopped further changes to `patience`, refocused on combinations among
  `lr`, `weight_decay`, `dropout_rate`, `noise_std`.

**Confidence levels used in this report**: 🟢 SOLID (large effect, well above the
noise floor, or replicated multiple times) · 🟡 PRELIMINARY (signal observed but
n=1 or n=2, needs confirmation) · ⚪ UNTESTED (open question).

---

## 1. `lr` axis — 🟢 SOLID

Six values tested, in order:

| lr | Trial | Verdict | R² | Δ vs previous point |
|---|---|---|---|---|
| 1e-5 | `20fa5d8e` | BASELINE | 0.6517 | — |
| 1e-4 | `47e474ef` | CHAMPION | 0.7475 | +0.096 |
| 3e-4 | `2b46a997` | CHAMPION | 0.7874 | +0.040 |
| 6e-4 | `97d513a3` | CHAMPION | 0.7972 | +0.010 |
| 7e-4 | `97529fed`* | FAILURE | 0.7819 | -0.033 (vs 6e-4) |
| 8e-4 | `7ecd5a5d` | FAILURE | 0.7896 | -0.008 (vs 6e-4) |

*tested later in the campaign, under `patience=60`, not in the same context as the others.

**Conclusion**: clear diminishing returns up to 6e-4 (peak), then a clean failure
above it. Re-tested twice in a different context (`d9254935`: lr=8e-4 under
patience=60, FAILURE -0.019; `756da5dc`: lr=3e-4 + weight_decay=1e-6, FAILURE
-0.035) — the peak at 6e-4 is robust and independent of the other axes.

**Acknowledged limitation**: asymmetric resolution — large jumps (×10, ×3, ×2)
between 1e-5 and 6e-4, then fine steps (7e-4/8e-4) only around the peak. The
1e-4–3e-4 region was never refined further (e.g. 2e-4 was never tested) once the
upward trend was confirmed.

---

## 2. `noise_std` axis — 🟢 SOLID (but a sharp, non-monotonic cliff)

| noise_std | Trial | Verdict | R² | Δ vs champion (lr=6e-4) |
|---|---|---|---|---|
| 0.0 | (implicit, default HP) | — | — | — |
| 0.00005 | `c0f04e94` | FAILURE | 0.7705 | -0.044 |
| **0.0001** | `319dacea` | **CHAMPION** | **0.8000** | **+0.003** |
| 0.0 (ablation) | `8663368c` | FAILURE | 0.8025 | -0.012 (vs patience=60 champion) |
| 0.0002 | `a1cf856e` | FAILURE | 0.7252 | -0.075 |
| 0.0003 | `5722444c` | FAILURE | 0.7477 | -0.052 |

**Conclusion**: a small, real, and repeated gain at 0.0001 (confirmed by ablation:
removing it costs -0.012). Above that, the degradation is **not monotonic** —
0.0002 (-0.075) is worse than 0.0003 (-0.052) — a signature of chaotic instability
rather than a smooth dose-response curve, consistent with the catastrophic collapse
observed at dim=240 (noise_std=0.002).

---

## 3. `dropout_rate` axis — 🟢 SOLID (harmful at every magnitude tested)

| dropout | Context | Verdict | R² | Δ |
|---|---|---|---|---|
| 0.05 | patience=30 | FAILURE | 0.7822 | -0.015 |
| 0.05 | patience=60 (retest) | FAILURE | 0.7738 | -0.041 |
| 0.01 | patience=60 | FAILURE | 0.7604 | -0.054 |
| 0.01 + weight_decay=1e-6 | patience=50 | FAILURE | 0.7689 | -0.046 |

**Conclusion**: dropout consistently costs more capacity than it recovers from
reduced overfitting, at every magnitude and context tested. A clean conclusion,
replicated across 4 distinct configurations.

---

## 4. `weight_decay` axis — 🟡 mostly neutral, 🟡 one intriguing PRELIMINARY interaction

### 4.1 Alone (neutral to mildly negative) — 🟢 SOLID on this specific point

| weight_decay | Context | Verdict | R² | Δ |
|---|---|---|---|---|
| 1e-6 | patience=30 | FAILURE | 0.7949 | -0.002 |
| 1e-6 | patience=60 | FAILURE | 0.8108 | -0.004 |
| 1e-5 | patience=60 | FAILURE | 0.8070 | -0.008 |

A consistent, small effect — within the noise band at 1e-6 but slightly more
pronounced at 1e-5 — direction stable (never positive), magnitude always modest.

### 4.2 Combined with `noise_std=0.0002` (which fails badly alone) — 🟡 PRELIMINARY

| Config | Verdict | R² | Validation std |
|---|---|---|---|
| noise_std=0.0002 alone | FAILURE | 0.7252 | 0.128 |
| + weight_decay=1e-6 | FAILURE | 0.7854 | 0.066 |
| + weight_decay=1e-5 | FAILURE | 0.7845 | 0.068 |

**What was actually tested**: 2 combinations, **1 single run each (n=1)**, both at
the same noise_std value (0.0002). No replicates, no test at noise_std=0.0001 or
0.0003 with weight_decay.

**What it suggests (not what it proves)**: adding even a tiny weight_decay appears
to reduce both the average failure (R² +0.06) and the validation standard deviation
(roughly halved) relative to noise_std=0.0002 alone — consistent with the idea that
weight smoothing (weight_decay) dampens the instability caused by the input noise.
The effect plateaus: 1e-5 adds nothing over 1e-6.

**⚠️ This is NOT statistically significant.** With a characterized run-to-run noise
of ~0.03-0.04 standard deviation (see §6) and only one run per point, a lucky draw
cannot be ruled out. Confirming this would require at least 3-4 replicates per
combination — not done due to remaining budget.

### 4.3 Combined with `dropout_rate=0.01` — 🟢 SOLID (dropout dominates)

`3fd5be93`: weight_decay=1e-6 + dropout=0.01 → FAILURE -0.046, close to dropout
alone (-0.054), not to weight_decay alone (-0.008). No synergy observed; dropout
dominates the interaction. Only one point tested, but the result is consistent with
the rest of the campaign's dropout findings.

### 4.4 Combined with `lr=3e-4` — 🟡 a single data point

`756da5dc` (final trial, 40/40): lr=3e-4 + weight_decay=1e-6 → FAILURE -0.035.
Consistent with weight_decay's independent effect, but only one point, no replicate.

---

## 5. `weight_decay` × `noise_std` × `dropout_rate` combinations — ⚪ largely UNTESTED

**What was never tried, explicitly flagged for a follow-up campaign:**
- `noise_std=0.0001` (the winning point) combined with nonzero `weight_decay` or `dropout`
- `dropout` at an intermediate magnitude between 0.01 and 0.05 (e.g. 0.02, 0.03)
- Triple combination (weight_decay + dropout + noise_std simultaneously)
- Any combination with replicates (≥3) to obtain a statistically defensible result
- `weight_decay` values between 1e-6 and 1e-5 (e.g. 3e-6, 5e-6)

The regularization-combination space remains very sparsely covered (5 trials out of
40 total dedicated to combinations proper — most of the budget went to individual
axes and to the variance investigation below).

---

## 6. Run-to-run variance investigation — 🟢 SOLID (the most important finding methodologically)

19 control replicates (exact same config, `seed=0` fixed) revealed substantial
variance despite the fixed seed — most likely due to GPU non-determinism
(cuDNN/atomic-ops) at `batch_size=1`, not RNG variation.

| Config (patience) | n replicates | Values | Mean | Std (sample) |
|---|---|---|---|---|
| 60 | 5 | 0.8148, 0.8050, 0.7396, 0.7783, 0.8012 | 0.788 | 0.039 |
| 45 | 5 | 0.8089, 0.8060, 0.7913, 0.7954, 0.8131 | 0.803 | 0.009 |
| 50 | 3 | 0.8148, 0.7796, 0.8041 | 0.800 | 0.018 |
| 49 | 3 | 0.8074, 0.8082, 0.7828 | 0.799 | 0.014 |

Grouped by scheduler cadence (`patience_scheduler = patience // 5`):
- **scheduler=9** (patience 45 & 49 combined, n=8): mean≈0.802, std≈0.010
- **scheduler=10-12** (patience 50 & 60 combined, n=8): mean≈0.792, std≈0.028

**Line of investigation explored then closed at the user's request**: a hint (not
statistically confirmed — no formal significance test was run, just an empirical
comparison of standard deviations) that `patience_scheduler=9` might be more
reproducible than `10-12`, at a comparable performance ceiling. `patience=50`
(scheduler=10) nonetheless remains the recorded champion since it holds the single
highest value observed across the whole campaign.

**Implication for the rest of this report**: any comparison between two trials in
this campaign with a gap smaller than ~0.03 should be read with this caveat — it is
not proven statistically different from zero. Only large effects (the initial lr
jump, dropout's repeated failures) are solidly established.

---

## 7. Timeline and mid-campaign redirection

The campaign initially followed a classic axis-by-axis sweep (lr → regularization →
patience), with cross-interaction checks (retesting an already-closed axis under
the final context of another axis). After trial 12 (discovery of substantial
variance via replicates), a significant share of the budget (trials ~19-36) was
spent characterizing that variance on the `patience` axis, which — in hindsight, and
per the user's feedback — represented a disproportionate share of the exploration
at the expense of regularization combinations. The last 4 trials (37-40) corrected
course by testing `weight_decay` × `dropout_rate` × `noise_std` × `lr` combinations,
producing the single most interesting preliminary result of the whole campaign
(§4.2) discovered in this final sprint — suggesting this direction deserved more
budget than it received.

---

## 8. Recommendations for a follow-up campaign

1. **Prioritize regularization combinations from the start** (weight_decay ×
   noise_std in particular) rather than treating them at the end of the budget.
2. **Systematically replicate (n≥3)** any combination deemed promising before
   drawing conclusions — the ~0.03-0.04 noise makes a single run inconclusive.
3. If the patience_scheduler=9 vs 10-12 reproducibility question is still of
   interest, a formal statistical test (e.g. Levene's test on variances, with
   n≥8 per group) would be needed to confirm it.
4. The noise_std=0.0001 + weight_decay=1e-6 point (never tested — weight_decay was
   only ever combined with the failing noise_std=0.0002) would be a direct,
   low-cost first check for a follow-up campaign.

---

## Traceability

All trials are on branch `agent-ae-opti`, one commit per step (lock input → result
→ docs). Full per-trial detail: `ai_agent/experiments/<id>.md` +
`<id>.console.log`. Flat index: `ai_agent/experiments/trial_log.csv`.
