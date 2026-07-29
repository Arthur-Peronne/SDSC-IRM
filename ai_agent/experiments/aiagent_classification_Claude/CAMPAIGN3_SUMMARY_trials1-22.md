<!-- Standalone summary, written on user request to pause the campaign after trial 22.
     Informational only — NOT part of the driver's record-keeping (trial_log.csv remains the
     authoritative ledger). Safe to leave uncommitted or delete once read. -->

# Campaign 3 summary — trials 1-22 (paused 2026-07-29)

**Judge:** `classification_accuracy_val` (logistic regression on AE latent codes predicting ACDC
group), mean over 3 seeds. R² (`validation_R2_mean`) logged for context only, not decisional.

**Budget used:** 22/25 trials (all recorded, including 1 mechanical failure). 3 trials remain if the
campaign resumes.

## Champion: `ac5057cf` — `AE3dAsymResSeparableV2SELateEnc3Only`
- `classification_accuracy_val = 0.7167` (+0.0167 vs. the prior tied champion)
- `validation_R2_mean = 0.7432` — best reconstruction of the whole campaign
- Fewer parameters than every other kept trial (1,128,969)
- Architecture: `AE3dAsymResSeparableV2` encoder/decoder (asymmetric residual separable convs,
  anisotropic pooling) with a single SE (squeeze-excitation) channel gate on `enc3` only (32
  channels) — no gate on `enc4`, no gate on `enc1`/`enc2`. `bottleneck_conv` is the plain two-conv
  `InstanceNorm3d` stack, unchanged from the original baseline. No skip connections (bottleneck rule
  respected throughout).
- **Caveat for the record:** per-seed accuracy was 0.750/0.750/0.650 — two seeds tied this
  campaign's best-ever single-seed score, one seed was notably weaker. This "two strong, one weak"
  pattern recurred in several trials built on this same base (`b02b8293`, `0afac601`, `582021d1`,
  `94dfcb9f`), suggesting part of `ac5057cf`'s margin over the previous champion may reflect favorable
  seed variance at `n_train=100`, not a large mechanistic gain. Treat the +0.0167 delta as real but
  modest, not a decisive win — consistent with `program.md`'s documented noise floor (~0.03-0.04 std).

## What was tried, grouped by mechanism (25 named architectures + 2 HP-only trials tested this
campaign; see `trial_log.csv` for the full ledger with exact deltas)

**SE (channel-gating) placement — the one mechanism with a clean track record.** Ablation grid now
complete: neither (BASELINE, 0.6667) < se4-only (FAILURE, 0.675) < full se1-4 (CHAMPION, 0.6917) <
se3+se4 (CHAMPION, 0.700) < **se3-only (CHAMPION, 0.7167)**. Se4 was mildly counterproductive, not
just redundant. SE gate capacity refinement (`reduction=8` instead of 16) was tried on top of
se3-only and failed (0.675) — the narrow 2-unit gate was not under-parameterized.

**Bottleneck (`bottleneck_conv`) — fully explored, no net win.**
Capacity reduction (single conv instead of two): FAILURE (0.65). Normalization
(`InstanceNorm3d`→`GroupNorm`): near-miss FAILURE (0.6917, but this campaign's then-best R²=0.742).
Residual/shortcut add (reusing `ResConv3DBlock`): FAILURE (0.6667). SE gate on the 128-channel
bottleneck output (`se5`): near-miss FAILURE (0.6917). None of these transferred when later fused with
the se3-only champion either (GroupNorm fusion: FAILURE 0.6583, worse R² than either parent alone).

**Receptive field (dilation) — tried at 3 locations, none won.**
`enc4` dilation=2: statistical tie with champion (not a real gain — driver's strict-inequality
tie-break happened to land CHAMPION). `enc3` dilation=2: clearer FAILURE (0.6667). Fused with
se3-only: FAILURE (0.6833), visibly slower/less stable convergence than se3-only alone.

**Other architecture mechanisms tried, all FAILURE:** CBAM spatial+channel attention at the
bottleneck (0.6667); learnable strided-conv downsampling instead of MaxPool (0.6167); nonlinear
(2-layer) FC bottleneck (0.6417).

**Hyperparameters — both `lr` and `patience` now closed, `weight_decay`/`dropout_rate` partially
explored.** `lr` sweep (on champion arch, no code change): `5e-5` FAILURE (0.675, smoothest
convergence of the campaign but WIDENED seed-to-seed accuracy variance); `2e-4` FAILURE (0.5667, this
campaign's second-worst result, consistent across all 3 seeds) → `1e-4` (the inherited default) looks
like a genuine local optimum, not an accident. `patience` sweep: `35` (vs. 20) FAILURE (0.6917) — more
training time shifted where each seed's optimizer landed but did not systematically rescue the weaker
seed, evidence the seed variance is closer to inherent than a fixable schedule issue. Untested this
campaign: `weight_decay` and `dropout_rate` individually (only tested once, jointly, early in the
campaign: `weight_decay=1e-5, dropout_rate=0.1` → FAILURE 0.6583); `noise_std` (tested in the prior,
since-archived Campaign 2, not retested here).

## Recurring findings worth carrying into any future campaign
1. **R² and classification accuracy are only loosely coupled.** Several trials moved one without the
   other in either direction (best-ever R² with no accuracy gain: `2647e285`; accuracy collapse with
   R² barely moved: `498a57b2`). Don't assume an R² improvement predicts a classification improvement.
2. **Independently-validated changes do not reliably compose.** Both fusion attempts on the current
   champion (GroupNorm, dilation) failed even though each component was individually tied/near-miss
   elsewhere — always retest fusions directly rather than inferring compatibility.
3. **A "two strong seeds, one weak seed" pattern recurs across many trials at this `n_train=100`,
   3-seed setup** — worth treating as a property of the evaluation protocol, not purely of any given
   architecture, when judging close deltas.

## If the campaign resumes (3 trials remain)
Untested, plausible next directions: `weight_decay`/`dropout_rate` individually (not just jointly);
`noise_std` on the current (leaner) champion, since it was only tested on the old full-SE lineage in
Campaign 2; a genuinely new architectural mechanism not yet touched (e.g. changing `enc1`/`enc2`,
kernel size instead of dilation, or a differently-scoped attention mechanism). Given the small budget
left, prioritize whichever direction has the clearest mechanistic rationale over further refinement of
`ac5057cf`, per `program.md`'s "not random search" guidance.
