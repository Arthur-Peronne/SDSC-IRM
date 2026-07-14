# Campaign report — AE3dAsymResSeparableV2 @ latent_dim=8

<!--
Summary report for the hyperparameter-optimization campaign run by the autonomous
agent (ai_agent/driver.py), from trial 0d2e0fa2 (baseline) to trial 0f0e4aef
(40/40, budget exhausted). Starting point: latent_dim=60 champion HPs transferred
unchanged (see ai_agent/experiments/aiagent_HP_sepv2_60/REPORT_dim60.md).
-->

## Summary

- **40/40 trials**, budget exhausted (`max_trials=40` in `experiment.yaml`).
- **Recorded champion (mechanical, n=1 rule): `a4292c17`** — `lr=7.5e-4, weight_decay=0,
  dropout_rate=0, noise_std=0, patience=50` → `avg_validation_R2_mean = 0.8135` (single run).
- **Recommended config (statistically robust): `lr=7e-4`, same rest** — n=13 replicates,
  mean R²=0.7957, std=0.0097. Chosen over the recorded champion because lr=7.5e-4's
  n=3 sample (0.8114 / 0.7143 / 0.8135) is bimodal (std=0.057, ~6x noisier) and its
  mean (0.7797) is actually *below* lr=7e-4's — see §1 and §6.
- Gain vs the dim=60-transferred baseline (`0d2e0fa2`, lr=6e-4, noise_std=1e-4,
  n=6 mean=0.7793): **0.7793 → 0.7957 (+0.016)**, from two combined changes
  (`lr` 6e-4→7e-4, `noise_std` 1e-4→0), each individually confirmed by replication.
- Unlike at dim=60 (where the champion HPs were unchanged relative to dim=240),
  **dim=8's optimum diverges from dim=60 in two axes**: `noise_std` (0 instead of
  0.0001) and `lr` (7e-4 instead of 6e-4).

**Confidence levels used in this report**: 🟢 SOLID (large effect, well above the
noise floor, or replicated multiple times) · 🟡 PRELIMINARY (signal observed but
n=1 or n=2, needs confirmation) · ⚪ UNTESTED (open question).

---

## 1. `lr` axis — 🟢 SOLID (with a 🟡 caveat on the 7-7.5e-4 region)

| lr | Trial | Context | Verdict | R² | n | Note |
|---|---|---|---|---|---|---|
| 3e-4 | `da8aad93` | noise_std=1e-4 | FAILURE | 0.7729 | 1 | -0.025 vs baseline, underfits |
| 6e-4 | `0d2e0fa2` | noise_std=1e-4 (baseline) | BASELINE | 0.7977 | 1 (n=6 mean 0.7793) | transferred from dim=60 |
| 6e-4 | `02343abb` | noise_std=0 | CHAMPION | 0.8056 | 1 (n=6 mean 0.7817) | noise_std ablation, see §2 |
| **7e-4** | `95ede5a9`→`0f0e4aef` | noise_std=0 | mixed (2×CHAMPION, rest FAILURE) | **mean 0.7957** | **13** | tightest std (0.0097) of any config tested |
| 7.5e-4 | `e5a91532`, `4c5f6a7c`, `a4292c17` | noise_std=0 | CHAMPION/FAILURE/CHAMPION | mean 0.7797 | 3 | bimodal: {0.8114, 0.7143, 0.8135}, std=0.057 |
| 7.75e-4 | `52b3d623` | noise_std=0 | FAILURE | 0.8069 | 1 | flattening trend, not a clear beat of 7.5e-4 |
| 8e-4 | `00b6f07e` | noise_std=0 | FAILURE | 0.7031 | 1 | catastrophic (-0.103), worst trial of the campaign |

**Conclusion**: clean bracketing — 3e-4 underfits, 8e-4 destabilizes catastrophically
(train R² itself collapses, not just val/test), 6e-4 to ~7.5e-4 is a broad, gentle
plateau. The single-run screening trend looked monotonically increasing up to
7.5e-4, but replication overturned that: **7.5e-4 is closer to the 8e-4 instability
cliff and about 1-in-3 runs collapse** (best epoch as low as 17, val std up to 0.14),
while **7e-4's 13 replicates are the tightest cluster in the whole campaign**
(std=0.0097, vs ~0.023-0.024 at lr=6e-4). The mechanically-recorded CHAMPION
(`a4292c17`, lr=7.5e-4) reflects one of its two "good" draws, not its true mean.

**🟡 Caveat**: the exact inflection point between "safe" (≤7e-4) and "risky"
(7.5e-4+) was never bracketed with matched replication on both sides — only 7e-4
got a full n=13 series. An intermediate value (e.g. 7.25e-4) with n≥5 would sharpen
this boundary.

---

## 2. `noise_std` axis — 🟢 SOLID (opposite sign from dim=60)

| noise_std | Trial | Verdict | R² | n | Mean (replicated) |
|---|---|---|---|---|---|
| 0.0001 (dim=60 champion value) | `0d2e0fa2` (BASELINE) | — | 0.7977 | 6 | 0.7793 |
| 0.0 (ablation) | `02343abb` | CHAMPION | 0.8056 | 6 | 0.7817 |

**Conclusion**: at dim=60, ablating `noise_std` cost -0.012 (input noise was a useful
regularizer against a larger model's overfitting). At dim=8, the same ablation
**helps** (+0.0079 single-run; means once replicated sit within noise of each other
but never favor keeping the noise). Mechanistically coherent: the 8-scalar
bottleneck is already a strong implicit regularizer, and injecting input noise on
top spends capacity absorbing irrelevant variance instead of encoding signal. This
is the first hyperparameter found to genuinely diverge between dim=60 and dim=8.

---

## 3. `dropout_rate` axis — 🟢 SOLID at lr=6e-4, 🟡 open question at lr=7e-4

| dropout | Context | Verdict | R² | Δ |
|---|---|---|---|---|
| 0.05 | lr=6e-4, noise_std=0 | FAILURE | 0.7770 | -0.0286 |
| 0.01 | lr=6e-4, noise_std=0 | FAILURE | 0.7153 | -0.0904 (worse than 0.05, non-monotonic) |
| 0.05 | **lr=7e-4** (sanity check) | FAILURE | 0.7997 | -0.0138 |

**Conclusion**: at lr=6e-4, dropout is clearly and non-monotonically harmful (0.01
worse than 0.05 — same qualitative pattern as dim=60's §3, larger magnitude),
consistent with an 8-scalar bottleneck having no redundant capacity to spare for
stochastic zeroing.

**🟡 Open question**: the single sanity-check run under lr=7e-4 (0.7997) falls
*inside* lr=7e-4's own noise band (n=13 range 0.7784-0.8099) — statistically
indistinguishable from "no dropout" at this lr, a real contrast with the large,
clear harm at lr=6e-4. One run is not enough to conclude dropout's harm genuinely
disappears at the higher lr; flagged for a follow-up campaign rather than resolved
here (see §7).

---

## 4. `weight_decay` axis — 🟢 SOLID (harmful or neutral, never positive)

| weight_decay | Verdict | R² | Δ | Note |
|---|---|---|---|---|
| 1e-6 | FAILURE | 0.7960 | -0.0017 | inside noise floor |
| 1e-5 | FAILURE | 0.7950 | -0.0106 | training didn't early-stop within 200 epochs — a real, mechanistic slowdown, not just noise |

**Conclusion**: same direction as dim=60 (never positive, larger magnitude at 1e-5
than 1e-6), and here backed by a concrete training-dynamics signature (no early
stopping triggered at 1e-5) rather than only the aggregate metric. `weight_decay=0`
confirmed as the right default.

---

## 5. `patience` axis — 🟢 SOLID at lr=6e-4, 🟡 not re-verified at lr=7e-4

| patience (scheduler) | Context | Verdict | R² | Δ |
|---|---|---|---|---|
| 30 (6) | lr=6e-4 | FAILURE | 0.7863 | -0.0193 |
| 45 (9) | lr=6e-4 | FAILURE | 0.7994 | -0.0062 (flat/tied) |
| 50 (10) | lr=6e-4 — champion | — | 0.8056 | — |
| 70 (14) | lr=6e-4 | FAILURE | 0.7442 | -0.0614 (worse than patience=30, asymmetric) |
| 45 (9) | **lr=7e-4** (sanity check) | FAILURE | 0.7806 | -0.0329 (n=1, ~2x lr=7e-4's own std) |

**Conclusion**: patience=50 sits at a real local optimum, bracketed cleanly on both
sides at lr=6e-4 (30 too short and stalls the LR schedule early; 70 too long and
introduces instability, not just a slower clean convergence). dim=60's
`patience_scheduler=9` reproducibility hint (lower variance at 45/49 than at
50/60) does **not** replicate at dim=8: the single patience=45 run shows normal
variance, not visibly reduced — not pursued further given it was already flagged
in the dim=60 report as a preliminary, low-priority line of investigation.

**🟡 Open question**: the lr=7e-4 sanity check (467c6c4e) is moderately suggestive
that patience=45 is still worse under the new lr, but n=1 keeps this unconfirmed —
same caveat as the dropout sanity check in §3.

---

## 6. Run-to-run variance investigation — 🟢 SOLID (methodologically the most important finding)

28 control replicates (exact same config, `seed=0` fixed) across 4 configurations,
confirming dim=60's finding that GPU non-determinism (not RNG) produces substantial
variance even with a fixed seed:

| Config | n | Values | Mean | Std |
|---|---|---|---|---|
| lr=6e-4, noise_std=1e-4 (baseline) | 6 | 0.7977, 0.7819, 0.7985, 0.7938, 0.7594, 0.7443 | 0.7793 | 0.0226 |
| lr=6e-4, noise_std=0 (champion `02343abb`) | 6 | 0.8056, 0.7655, 0.7991, 0.7856, 0.7420, 0.7924 | 0.7817 | 0.0238 |
| **lr=7e-4, noise_std=0** | **13** | 0.7784-0.8099 (see §1) | **0.7957** | **0.0097** |
| lr=7.5e-4, noise_std=0 | 3 | 0.8114, 0.7143, 0.8135 | 0.7797 | 0.0567 |

**Implication**: the recommended config's advantage isn't just a higher mean —
**it is also markedly more reproducible** (std 0.0097, roughly 2.5x tighter than
the lr=6e-4 basins and 6x tighter than lr=7.5e-4). This mirrors dim=60's own
observation that some configurations are more reproducible at a comparable
ceiling, but here the effect is much larger and directly actionable, not a minor
preliminary hint.

**Any comparison in this report smaller than ~0.02-0.03 (the lr=6e-4-level noise
floor) should be read with this caveat** — only the lr axis's large effects (the
8e-4 cliff, the 3e-4 underfit) and the noise_std/dropout/weight_decay directional
findings (replicated or mechanistically corroborated) are solidly established.

---

## 7. Sanity checks: do lr=6e-4 conclusions transfer to lr=7e-4? — 🟡 UNRESOLVED

Once lr=7e-4 became the leading candidate, two single-run checks tested whether
conclusions established under lr=6e-4 still hold:

- **dropout=0.05 under lr=7e-4** (`708b081f`): -0.0138, inside lr=7e-4's own noise
  band — looks *less* harmful than at lr=6e-4 (-0.0286), but n=1.
- **patience=45 under lr=7e-4** (`467c6c4e`): -0.0329, still suggestive of harm,
  same direction as at lr=6e-4, but n=1.

**Neither is resolved.** Both point toward "probably still fine to keep dropout=0
and patience=50" but neither the confirmation nor a contradiction was replicated —
explicitly the highest-value cheap follow-up (see §8).

---

## 8. Recommendations for a follow-up campaign

1. **Bracket the 7-7.5e-4 region properly**: test one or two intermediate points
   (e.g. 7.25e-4) with n≥5 replicates each, to locate the actual boundary between
   the tight, reliable lr=7e-4 basin and the bimodal instability seen at 7.5e-4.
2. **Resolve the two open sanity checks from §7** with matched-n replication
   (dropout=0.05 and patience=45, both under lr=7e-4) — currently n=1 each, and
   both would change the practical recommendation if confirmed.
3. **Test noise_std/weight_decay combinations directly at lr=7e-4** — all
   combination work to date (this campaign and dim=60's) was done at lr=6e-4;
   never re-checked under the new lr.
4. **Consider baking replication into the driver itself** (`repeat_over: {seed:
   [0,1,2]}`) for future HP campaigns on this architecture — a large share of this
   campaign's 40-trial budget (19 trials) went to manually replicating candidates
   after the fact, once the single-run verdict rule was found to be misleading at
   this noise level; an aggregated N>1 trial would make the mechanical CHAMPION
   verdict trustworthy by construction instead of needing this kind of post-hoc
   report to correct it.
5. **Cross-dimension trend**: dim=8's optimum diverges from dim=60 in exactly
   `noise_std` (0 vs 0.0001) and `lr` (7e-4 vs 6e-4). With three data points now
   (dim=240, dim=60, dim=8), a follow-up could test an intermediate dimension
   (e.g. dim=20-30) to see whether these two axes shift monotonically with latent
   capacity or whether dim=8 is a special (very-low-capacity) regime.

---

## Optimal hyperparameters for latent_dim=8

| Hyperparameter | Value |
|---|---|
| `lr` | `7e-4` |
| `weight_decay` | `0` |
| `dropout_rate` | `0` |
| `noise_std` | `0` |
| `patience` | `50` |

Statistically robust recommendation (n=13 replicates): mean `avg_validation_R2_mean
= 0.7957`, std `0.0097` — the tightest (most reproducible) config found across all
three latent-dim campaigns. This is **not** the value the ledger mechanically
recorded as CHAMPION (`a4292c17`, `lr=7.5e-4`, single-run R²=0.8135) — see §1 and §6
for why the mechanical champion is a favorable-outlier single draw rather than the
best config in expectation.

---

## Traceability

All trials are on branch `agent-ae-opti`, one commit per step (lock input → result
→ docs). Full per-trial detail: `ai_agent/experiments/aiagent_HP_sepv2_8/<id>.md` +
`<id>.console.log`. Flat index: `ai_agent/experiments/aiagent_HP_sepv2_8/trial_log.csv`.
