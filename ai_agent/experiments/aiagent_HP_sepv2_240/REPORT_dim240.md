# Campaign report — AE3dAsymResSeparableV2 @ latent_dim=240

<!--
Summary report for the hyperparameter-optimization campaign run by the autonomous
agent (ai_agent/driver.py), from trial 185cf97f (baseline) to trial 93b15034
(36 trials, stopped early by user decision — see § Timeline). First campaign
under the new driver protocol; dim=60 (aiagent_HP_sepv2_60/) and dim=8
(aiagent_HP_sepv2_8/) followed it, both starting from HPs carried over from here.
-->

## Summary

- **36 trials run**, out of a budget extended mid-campaign from `max_trials=30` to
  `50` (user-approved). The campaign was archived/closed at 36 by user decision, not
  by the driver reaching the budget — see § Timeline for why.
- **Recorded champion (mechanical, n=1 rule): `3e07b08d`** — `lr=8e-4, weight_decay=0,
  dropout_rate=0.05, noise_std=0, patience=30` → `avg_validation_R2_mean = 0.8277`
  (single run), vs baseline `185cf97f` (lr=5e-4, all regularizers off) at 0.8075.
- **⚠️ This is the campaign's central, hard-won finding: the improvement does not
  hold up under replication.** 5 independent runs of the exact champion config
  gave `{0.8277, 0.8031, 0.8048, 0.7799, 0.7643}` — mean 0.7960, std 0.0245 — a
  mean that is *at or below* the single-run baseline (0.8075). The recorded
  0.8277 looks like a favorable outlier, not a representative value. See §6.
- Every attempt to stack a further regularizer (weight_decay, noise_std, extra
  patience) onto the champion was neutral-to-redundant — see §7.
- **Practical conclusion, stated plainly by the campaign's own final trial**: this
  campaign's single-run-per-trial protocol was not able to reliably detect a real
  improvement over the baseline at `latent_dim=240`. See § Optimal hyperparameters
  for how this shapes the recommendation.

**Confidence levels used in this report**: 🟢 SOLID (large effect, well above the
noise floor, or replicated multiple times) · 🟡 PRELIMINARY (signal observed but
n=1 or n=2, needs confirmation) · ⚪ UNTESTED (open question).

---

## 1. `lr` axis — 🟢 SOLID direction, 🟡 exact peak unresolved

| lr | Context | Verdict | R² | Δ |
|---|---|---|---|---|
| 2e-4 | alone, baseline dropout/wd/noise off | FAILURE | 0.7888 | -0.0186 |
| 5e-4 | baseline | BASELINE | 0.8075 | — |
| 6.5e-4 | alone | FAILURE | 0.7707 | -0.0367 |
| 5e-4 + dropout=0.05 | (old lr, champion's dropout) | FAILURE | 0.8083 | -0.0194 (vs champion) |
| 6e-4 + dropout=0.05 | | FAILURE | 0.8018 | -0.0260 (vs champion); replicate: 0.7920 |
| 7e-4 + dropout=0.05 | | FAILURE | 0.7446 | -0.0832 — a clear valley between 6e-4 and 8e-4 |
| **8e-4** | alone | FAILURE (near-tie) | 0.8066 | -0.0009 |
| **8e-4 + dropout=0.05** | **champion pairing** | CHAMPION | 0.8277 | — (but see §6) |
| 9e-4 + dropout=0.05 | | FAILURE | 0.7561 | -0.0716 |
| 1e-3 | alone | FAILURE | 0.7519 | -0.0555 (instability) |
| 1e-3 + dropout=0.05 | (testing if dropout stabilizes it) | FAILURE | 0.7980 | -0.0297 — still fails, though less badly |

**Conclusion**: `lr` alone plateaus around 5e-4 to 8e-4 (all near-tied, single-run
noise-bound), then degrades toward 1e-3 (instability). The striking finding is
that **lr and dropout are not independent**: `lr=7e-4 + dropout=0.05` is a sharp,
deep valley (-0.083) sitting *between* two much better points (6e-4 and 8e-4 with
the same dropout) — a non-monotonic interaction, not a smooth curve. `lr=8e-4`
alone nearly tied the baseline with a better raw val_loss and faster convergence,
which motivated fusing it with dropout (§2) into the recorded champion.

---

## 2. `dropout_rate` axis — 🟡 real mechanism, but magnitude/reliability unresolved

| dropout | lr | Verdict | R² | Δ |
|---|---|---|---|---|
| 0.15 | 5e-4 (baseline lr) | FAILURE | 0.7993 | -0.0082 |
| 0.03 | 8e-4 | FAILURE | 0.7608 | -0.0669 |
| 0.04 | 8e-4 | FAILURE (near-tie) | 0.8196 | -0.0081; **replicate: 0.7838 (-0.0439)** |
| **0.05** | **8e-4** | **CHAMPION** | **0.8277** | pairing that defines the champion |
| 0.045 | 8e-4 | FAILURE | 0.7912 | -0.0365 |
| 0.06 | 8e-4 | FAILURE | 0.7946 | -0.0331 |
| 0.08 | 8e-4 | FAILURE | 0.7561 | -0.0716 |
| 0.06 | 6e-4 (transfer check) | FAILURE | 0.7845 | -0.0432 |
| 0.06 | 5e-4 (transfer check) | FAILURE | 0.7913 | -0.0364 |
| 0.08 | 5e-4 (transfer check) | FAILURE | 0.7978 | -0.0299 |

**Conclusion**: the champion trial's own diagnostics (best val_loss of the whole
campaign, lowest validation std, narrower train/val gap, longer productive
training) made a mechanistically coherent case that `dropout=0.05` at `lr=8e-4`
specifically unlocks useful regularization that neither factor achieves alone —
higher lr converges faster, leaving more "room" in the training budget for a small
dropout dose to pay off before early stopping. **However, the 0.04 replicate
(0.8196 → 0.7838, a 0.036 swing on a "near-tying" point) already hints the fine
structure around the champion is noise-dominated**, a warning sign the campaign's
later champion-replicate investigation (§6) confirmed at the champion's own exact
point. Dropout does not transfer as a universal win at other lr values (5e-4,
6e-4) — the effect is specific to the lr=8e-4 pairing, not a standalone dropout
recommendation.

---

## 3. `noise_std` axis — 🟢 SOLID (catastrophic at the naively-borrowed magnitude)

| noise_std | Context | Verdict | R² | Δ |
|---|---|---|---|---|
| 0.002 (Optuna-borrowed value) | alone | FAILURE | 0.6223 | **-0.1851**, worst single-axis result of the campaign |
| 0.0002 | alone | FAILURE | 0.7900 | -0.0174 |
| 0.0001 | on champion (lr=8e-4+dropout=0.05) | FAILURE | 0.7976 | -0.0301 |

**Conclusion**: borrowing `noise_std=0.002` from an Optuna search tuned under
different conditions (batch_size=1 here means no batch-averaging cancels the
per-step noise) was decisively wrong — training itself degraded (train R² fell to
0.65), not just generalization. At 10x smaller magnitudes, the effect is neutral
to mildly negative and redundant once dropout is already active on the champion —
consistent with the "regularization budget" reading in §7.

---

## 4. `weight_decay` axis — 🟢 SOLID (neutral-to-negative, never clearly positive)

| weight_decay | Context | Verdict | R² | Δ |
|---|---|---|---|---|
| 1e-5 | alone, lr=5e-4 | FAILURE | 0.7782 | -0.0293 |
| 1e-6 | on champion (lr=8e-4+dropout=0.05) | FAILURE | 0.8110 | -0.0167 (inside noise band) |
| 1e-6 | + lr=5e-4+dropout=0.05 | FAILURE | 0.8022 | -0.0255 |

**Conclusion**: same story as at dim=60/dim=8 — weight_decay is never the axis
that moves this architecture forward, at any magnitude or combination tested here.

---

## 5. `patience` axis — 🟢 SOLID (a hard floor, ceiling not clearly beneficial)

| patience | Context | Verdict | R² | Δ |
|---|---|---|---|---|
| 15 | on champion | FAILURE | 0.6910 | **-0.1367**, second-worst result of the campaign — stops mid-descent |
| 30 (default) | — | — | — | never itself varied at the baseline's own lr |
| 45 | on champion | FAILURE | 0.8082 | -0.0196 (neutral, inside noise) |
| 50 | alone, lr=5e-4 | FAILURE | 0.7799 | -0.0276 |

**Conclusion**: patience=15 is a clean, large FAILURE — this architecture's useful
improvements at dim=240 commonly arrive at best-epochs of 40-100+, so a 15-epoch
plateau stops training while still in early descent. Above the default (45), no
clear gain — patience=30 stands as a reasonable middle ground, not a bottleneck.

---

## 6. Champion reliability investigation — 🟢 SOLID (the campaign's central finding)

5 independent runs of the exact recorded-champion config (`lr=8e-4,
dropout_rate=0.05`, all else at baseline defaults, `seed=0` fixed):

| Trial | R² |
|---|---|
| `3e07b08d` (the one the ledger recorded as CHAMPION) | 0.8277 |
| `59ff727f` | 0.8031 |
| `f5873a7c` | 0.8048 |
| `bb2f2c0f` | 0.7799 |
| `93b15034` | 0.7643 |

**Mean = 0.7960, std = 0.0245.** The running mean *dropped with every additional
replicate* (0.828 → ~0.815 → ~0.812 → ~0.804 → 0.796), and by n=5 it sits at or
slightly below the single-run baseline (0.8075). The single highest value —
exactly the one the driver's fixed, single-run verdict rule uses to decide
CHAMPION — looks like a favorable outlier from a config whose true expectation is
roughly tied with (not better than) doing nothing.

**Implication**: unlike dim=60 (where the champion's gain was large enough to
survive this scrutiny) and dim=8 (where a 13-replicate mean confirmed a real,
if modest, gain), **dim=240's recorded champion is the one case across all three
latent-dim campaigns where replication overturned rather than confirmed the
single-run verdict.** Any single-run delta in this report smaller than roughly
±0.025 (this campaign's measured noise floor) should be read with that in mind.

---

## 7. Regularization-stacking on the champion — 🟢 SOLID ("regularization budget" reading)

Every attempt to add a second regularizer on top of the champion's `lr=8e-4 +
dropout=0.05` was neutral-to-mildly-negative, never additive:

| Addition | Verdict | R² | Δ vs champion |
|---|---|---|---|
| + weight_decay=1e-6 | FAILURE | 0.8110 | -0.0167 |
| + noise_std=0.0001 | FAILURE | 0.7976 | -0.0301 |
| + weight_decay=1e-6 AND noise_std=0.0001 | FAILURE | 0.8032 | -0.0245 (no synergy vs either alone) |
| + patience=45 | FAILURE | 0.8082 | -0.0196 |

**Conclusion**: dropout=0.05 at lr=8e-4 appears to exhaust whatever regularization
this architecture/dataset combination can usefully absorb at `latent_dim=240` —
consistent across every mechanistically distinct addition tried (deterministic L2,
input-space noise, longer patience), alone and combined. Given §6, this reading
should itself be treated cautiously: if the champion's own edge over baseline is
largely noise, "no further gain from stacking" may partly reflect that there was
little real edge to add to in the first place.

---

## 8. Timeline and why the campaign stopped at 36/50

The campaign followed a clean single-axis-around-baseline sweep first (trials 1-9:
dropout, weight_decay, lr down, noise_std, patience, noise_std smaller, lr up,
lr further up, lr fine-tune) — 3 FAILUREs establishing that no individual axis beat
the baseline. Trial 10 (`3e07b08d`) then fused two promising-but-individually-
insufficient leads (lr=8e-4's near-tie, dropout's plausible-but-untested-at-smaller-
dose mechanism) into the recorded champion. The following ~20 trials mapped its
neighborhood exhaustively: dropout fine-tuning (0.03-0.08), lr fine-tuning with
dropout fixed (5e-4 to 1e-3), regularizer stacking (§7), and — critically —
4 direct replicates of the champion itself, undertaken specifically because the
early neighborhood trials showed suspiciously large scatter (e.g. the dropout=0.04
near-tie swinging by 0.036 on a single replicate). Those 4 replicates revealed the
champion's true instability (§6), and the campaign was archived at 36/50 trials —
a deliberate early stop once the central open question ("is the champion real?")
had a clear, if disappointing, answer, rather than continuing to explore a
neighborhood whose apparent structure was substantially noise.

---

## 9. Recommendations for a follow-up campaign

1. **Re-run this campaign with `repeat_over: {seed: [0,1,2]}` baked into the
   driver from the start** — the single biggest lesson here is that a single-run
   verdict at this noise level (~0.025 std) is not trustworthy for deciding
   CHAMPION at `latent_dim=240`; this campaign only found that out empirically,
   9 trials deep into exploring a champion neighborhood that mostly wasn't real.
2. **The lr=7e-4+dropout=0.05 valley (-0.083, between two much better points at
   6e-4 and 8e-4) deserves its own replication** — it is currently n=1 and either
   a real sharp interaction effect or itself a noise artifact; unresolved either
   way given §6's demonstrated scatter.
3. **Re-evaluate whether dropout=0.05 (or any dropout) is genuinely beneficial at
   this dim** given the champion's mean is now statistically tied with the
   (also single-run, unreplicated) baseline — a matched-n replication of the
   *baseline* itself (never done here) is the missing piece to make this
   comparison fair.
4. **Cross-dimension consistency check**: dim=60 and dim=8 both found
   dropout/weight_decay/extra noise_std harmful or neutral and converged on a
   confidently non-zero HP change (lr shift, noise_std removal) with multi-run
   support. dim=240 is the outlier where the "improvement" didn't survive
   replication — worth checking whether this reflects a genuine property of the
   larger bottleneck (more capacity → more sensitive to the specific noise
   realization) or simply that this campaign, being first, replicated less
   systematically than the later two.

---

## Optimal hyperparameters for latent_dim=240

**⚠️ Read this alongside §6 before using these values**: the recorded ledger
champion's edge over the baseline did not survive replication (n=5 mean 0.7960
vs. the single-run baseline's 0.8075) — this is the one latent-dim campaign of
the three where the mechanical verdict is *not* backed by a confirmed improvement.

| Hyperparameter | Recorded champion (`3e07b08d`) value |
|---|---|
| `lr` | `8e-4` |
| `weight_decay` | `0` |
| `dropout_rate` | `0.05` |
| `noise_std` | `0` |
| `patience` | `30` |

`avg_validation_R2_mean` (single run, ledger value): **0.8277** — but the honest
estimate, from 5 replicates of this exact config, is **mean 0.7960 (std 0.0245)**,
statistically indistinguishable from simply using the un-regularized baseline
(`lr=5e-4`, all four other HPs at 0, `patience=30`, single-run R²=0.8075). Use
these values as a starting point for a follow-up campaign, not as a settled
recommendation.

---

## Traceability

All trials are on branch `agent-ae-opti`, one commit per step (lock input → result
→ docs). Full per-trial detail: `ai_agent/experiments/aiagent_HP_sepv2_240/<id>.md`
+ `<id>.console.log`. Flat index:
`ai_agent/experiments/aiagent_HP_sepv2_240/trial_log.csv`.
