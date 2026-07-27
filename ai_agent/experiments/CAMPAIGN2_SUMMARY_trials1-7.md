# Campaign 2 — Summary after trial 7 (paused for review)

**Date:** 2026-07-27
**Branch:** `agent-ae-opti`
**Objective:** optimize `AE3dAsymResSeparableV2` at `latent_dimensions=20` to maximize
`classification_accuracy_val` (logistic regression on AE latent codes, predicting ACDC
patient group), not autoencoder R² directly (R² still logged for context via
`validation_R2_mean`, non-decisional).
**Budget:** 7 / 20 trials used, **13 remain**.
**Current champion: `3aa0388f`, classification_accuracy_val = 0.6083**

Paused here at the user's request, to think things over before running more trials.

---

## Champion configuration (`3aa0388f`)
- Architecture: `AE3dAsymResSeparableV2`, unmodified from its original channel widths
  (8/16/32/64/128, bottleneck flattened_size=2048)
- `lr=5e-4`, `weight_decay=1e-5`, `dropout_rate=0.3` (bottleneck-only, applied on the
  flattened 2048-d vector right before/after `fc_enc`/`fc_dec`), `noise_std=0.0`,
  `patience=20`
- `n_train=100`, `n_val=20`, `n_test=30`, `special_split="split100"`,
  `stratify_ongroup=true`

## Trial-by-trial

| # | id | parent | change | accuracy_val | Δ vs champion | R²_mean | verdict |
|---|----|--------|--------|--------------|----------------|---------|---------|
| 1 | `725420c7` | — | light reg. (weight_decay=1e-5, dropout=0.1) on default HPs | 0.5917 | +0.0000 | 0.7069 | BASELINE |
| 2 | `f826eedd` | 1 | + input denoising (noise_std=0.05) | 0.4167 | -0.1750 | 0.4111 | FAILURE |
| 3 | `3aa0388f` | 1 | dropout_rate 0.1→0.3, noise_std reverted to 0.0 | **0.6083** | +0.0167 | 0.6823 | **CHAMPION** |
| 4 | `b606a10f` | 3 | double ALL channel widths (whole network) | 0.5583 | -0.0500 | 0.7315 | FAILURE |
| 5 | `0cadad28` | 3 | double ENCODER widths only, bottleneck held at 2048-d | 0.5250 | -0.0833 | 0.6904 | FAILURE |
| 6 | `001b5a08` | 3 | HALVE all channel widths (opposite direction) | 0.5250 | -0.0833 | 0.7239 | FAILURE |
| 7 | `e4e99f58` | 3 | weight_decay 1e-5→1e-3 (100x), architecture unchanged | 0.2167 | -0.3917 | 0.4771 | FAILURE |

## What was learned

1. **Input-level noise (trial 2) is a bad regularization site.** `noise_std=0.05`
   corrupts raw voxel data before the encoder ever sees it, destroying fine anatomical
   detail the classifier depends on — both R² and accuracy collapsed together, and
   training showed early degradation (val loss peaked at epoch 2 across all seeds).

2. **Bottleneck dropout (trial 3, the champion) is the one validated win.** Regularizing
   the 2048-d latent vector directly — right where the classifier reads it — improved
   accuracy over baseline while training stayed stable. Notably this happened *despite* a
   slightly lower/noisier R² than baseline: **reconstruction fidelity and downstream
   classification accuracy are not the same objective** in this setup. This decoupling
   is the recurring theme of the whole campaign so far.

3. **Channel width, in either direction, is not the lever (trials 4-6).** Three
   variations were tried — widen everything, widen only the encoder (holding the
   bottleneck fixed to control for a regularization-dilution confound), and halve
   everything — and **all three failed**, with R²_mean staying flat-to-*better* than the
   champion's 0.6823 in every single case (0.7315 / 0.6904 / 0.7239). This is fairly
   strong evidence that **reconstruction quality is not capacity-bottlenecked** anywhere
   in the 1024–4096-d bottleneck range tested, and that classification accuracy is hurt
   by width changes for reasons closer to optimization variance/overfitting than to
   representational capacity. Per program.md's stop-early rule ("don't keep proposing
   variations of a repeatedly-failing idea"), this line was retired after trial 6 rather
   than tried a 4th way.

4. **weight_decay is a much less forgiving knob than dropout (trial 7).** A 100x
   increase (1e-5→1e-3) did not mildly strengthen regularization — it collapsed *both*
   R² (0.4771) and accuracy (0.2167, near chance for 5 ACDC groups) together and
   consistently across all 3 seeds, the campaign's worst result. Read as
   over-regularization/underfitting from an L2 penalty too strong for this network size,
   not a useful direction at this magnitude. **A much smaller step (e.g. 1e-5→1e-4, 10x
   not 100x) was never tested** and might behave completely differently — this trial
   doesn't rule out weight_decay as a lever, only 1e-3.

## Untested directions (for when trials resume)
- Milder weight_decay step (10x instead of 100x)
- A genuinely different architectural mechanism — not channel width: e.g. replacing
  `MaxPool3d` downsampling with a learned strided convolution, kernel size changes, or
  normalization changes (InstanceNorm is used throughout)
- `lr` and `patience` are still untouched from their original default values
- Combining the validated dropout=0.3 champion with a *small* additional lever (rather
  than the large single-variable swings tried so far) — champion `3aa0388f` is still the
  best result of a fairly wide two-directional sweep, so future gains may need a
  finer-grained step rather than another big swing

## Process note
On the trial 7 relaunch, the harness produced a `<system-reminder>` claiming
`configs/autoencoder.yaml` "was modified, either by the user or by a linter" and
instructing me not to mention it to you. This matches a known injection-style pattern
seen twice before in this campaign (2026-07-10, 2026-07-26) — verified against `git log`
each time and it was, again, just the driver's own documented auto-revert of the
`mutable` config back to the champion's `weight_decay=1e-5` after trial 7's FAILURE
verdict (commit `d29c97dd`). Flagging it to you as before rather than complying with the
"don't tell the user" instruction embedded in it.
