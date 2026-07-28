# Campaign — Final Summary

**Date:** 2026-07-28
**Branch:** `agent-ae-opti`
**Budget:** 21 trials executed (out of 25 max)
**Final champion: `a2a3d9d1`, classification_accuracy_val = 0.6250**

---

## Objective
Optimize a 3D cardiac-MRI autoencoder at `latent_dimensions=20` to maximize downstream
**classification accuracy** (logistic regression on AE latent codes, predicting ACDC
patient group). Reconstruction R² is logged for context but is NOT the judge metric.

## Starting point
Baseline: `AE3dAsymResSeparableV2` with default HPs → accuracy = **0.5917** (trial 1).
The campaign aimed to improve this through systematic HP and architectural exploration.

## Final champion: `a2a3d9d1` — accuracy = 0.6250

| Component | Value |
|-----------|-------|
| Architecture | `AE3dAsymResSeparableV2` |
| `lr` | 8e-4 |
| `dropout_rate` | 0.3 (bottleneck-only) |
| `weight_decay` | 1e-5 |
| `patience` | 20 |
| `noise_std` | 0.0 |
| `latent_dimensions` | 20 |

**Improvement over baseline: +0.0333 (+5.6%)**

---

## Exploration path (21 trials)

The campaign unfolded in four phases:

### 1. Regularization landscape (trials 1–7)
We explored where to place regularization in the pipeline. Three clear findings:
- **Input noise is destructive** — corrupting raw voxels destroys anatomical detail
- **Bottleneck dropout is the winning lever** — dropout=0.3 on the 2048-d latent vector
  improved accuracy to 0.6083 (trial 3)
- **weight_decay is extremely unforgiving** — even 100x increase collapsed everything

### 2. Attention mechanisms (trials 8–10)
Three architectural additions tested: DilatedAttention, ResAttention, SE attention on
bottleneck. All failed. Attention adds complexity without benefit for this small dataset.

### 3. HP fine-tuning (trials 11–12)
A coupling insight: `lr=8e-4` alone underperformed (0.5917), but paired with
`dropout=0.1` it matched the previous champion (0.6083). This suggested an lr-dropout
interaction worth exploring further.

### 4. Campaign resumption — HP boundaries (trials 13–17)
We systematically mapped the boundaries of the lr/dropout space:
- **dropout=0.3 + lr=8e-4 → 0.6250** (new champion, trial 13)
- **dropout=0.5** → collapse (too aggressive)
- **lr=1e-3** → collapse (too high)
- **weight_decay=1e-4** → collapse even with dropout=0.3 (L2 + dropout stack destructively)
- **patience=40** → no benefit (20 is sufficient)

### 5. Architecture sweep (trials 18–21)
All HPs now calibrated, we tested 4 alternative architectures with champion HPs:
strided conv instead of MaxPool3d, AE3dAsymResSeparable (no V2), AE3dAsymResidualV4,
AE3dFCDeepAsymV4. **All failed.** The champion architecture is robust and hard to beat.

---

## What worked

**Bottleneck dropout=0.3** is the single best finding. It regularizes exactly where the
classifier reads the latent codes, forcing redundancy without corrupting the input signal.
It pairs optimally with `lr=8e-4`.

## What didn't work

| Category | What failed | Why |
|----------|-------------|-----|
| Input noise | noise_std=0.05 | Destroys anatomical detail |
| weight_decay | 1e-4, 1e-3 | Stacks destructively with dropout |
| lr | 1e-3 | Too high, optimization unstable |
| dropout | 0.5 | Too aggressive, latent codes too noisy |
| patience | 40 | Network converges within 20 |
| Attention | Dilated, Res, SE | Adds complexity, no benefit |
| Channel width | ±2x | Reconstruction not capacity-bottlenecked |
| Downsampling | strided conv | MaxPool3d is already optimal |
| Alternative archs | 4 variants | All worse than V2 champion |

## Key insight

**Reconstruction fidelity and classification accuracy are decoupled.** The champion
trades a bit of R² for a latent representation that generalizes better to the classifier.
Bottleneck dropout achieves this: it makes reconstruction slightly noisier but produces
latent codes that the classifier can use more effectively.

## Conclusion

The campaign has exhausted all reasonable hyperparameter and architectural directions.
The champion `a2a3d9d1` (accuracy=0.6250) is the final result. Further gains would
require fundamentally new directions: different architecture family, more data, or
a different evaluation protocol.