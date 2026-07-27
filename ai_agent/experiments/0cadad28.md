---
model_name: AE3dAsymResSeparableV2
summary: Widen ONLY the encoder path (1/8/16/32/64 -> 1/16/32/64/128); bottleneck_conv,
  flattened_size (2048) and decoder unchanged, isolating capacity from the bottleneck-dilution
  confound of FAILED trial b606a10f
parent: 3aa0388f
id: 0cadad28
status: completed
verdict: FAILURE
created_at: '2026-07-27T07:37:44+00:00'
metric:
  primary:
    name: classification_accuracy_val
    value: 0.525
    direction: maximize
---

# Trial 0cadad28 — AE3dAsymResSeparableV2 — FAILURE

<!-- ===== written BEFORE the run (agent) ===== -->

## Hypothesis
Trial b606a10f (FAILURE, -0.05 vs champion) doubled EVERY channel width including the
bottleneck (128->256, flattened 2048->4096) and, despite validation_R2_mean improving to
0.7315 (best of the campaign), classification_accuracy_val fell to 0.5583. My conclusion
there was that the enlarged bottleneck under the *same* dropout_rate=0.3 is
proportionally less regularized (roughly 2x the retained non-dropped units at the same
drop probability), letting the classifier overfit train-specific reconstruction detail
that doesn't generalize to ACDC-group discrimination. This trial isolates the capacity
variable from that confound: only the ENCODER path (enc1-enc4, before the bottleneck) is
widened, while `bottleneck_conv`, `final_down`, `feature_shape`/`flattened_size` (kept at
2048, identical to champion) and the entire decoder are UNCHANGED from champion 3aa0388f.
Mechanistically: if the champion's classification bottleneck was under-capacity because
the *encoder* couldn't extract enough fine-grained anatomical detail before spatial
downsampling discards it (my original trial-4 hypothesis), richer encoder features
should still help even when compressed back down to the same 2048-d bottleneck under the
same dropout=0.3 — the regularization strength relative to the decision-making
representation (the bottleneck the classifier reads) is held constant this time, so any
improvement (or lack thereof) is attributable to encoder capacity alone, not to diluted
regularization. I predict this recovers at least part of trial b606a10f's R2 gain while
avoiding its classification regression, since the classifier is exposed to the same
regularization strength as the validated champion.

## Implementation
In `src/models/ae_models.py`, class `AutoEncoder3D_AsymResSeparableV2`: only the encoder
blocks are widened — `enc1` 1->16 (was 1->8), `enc2` 16->32 (was 8->16), `enc3` 32->64
(was 16->32), `enc4` 64->128 (was 32->64). `bottleneck_conv`'s first conv now takes 128
channels in (matching the widened `enc4` output) but still outputs 128 (unchanged from
champion — this is the only place channel count does NOT double, by design). Everything
downstream is byte-for-byte identical to champion 3aa0388f: `bottleneck_conv`'s second
conv (128->128), `final_down` (128->128), `feature_shape`=(128,1,4,4),
`flattened_size`=2048, `fc_enc`/`fc_dec`, and the full decoder (`initial_up`, `dec1`
128->64, `dec2` 64->32, `dec3` 32->16, `dec4_conv` 16->8, `final_conv` 8->1). No skip
connections. In `configs/autoencoder.yaml` (only field touched): none — all HPs
(`lr=5e-4`, `weight_decay=1e-5`, `dropout_rate=0.3`, `noise_std=0.0`, `patience=20`) kept
identical to champion, isolating the encoder-capacity change alone.

<!-- ===== written AFTER the run ===== -->

## Results
- **accuracy_test per run:** 0: 0.600000 | 1: 0.525000 | 2: 0.450000
- **classification_accuracy_val:** 0.525000
- **delta_vs_champion** (display only): -0.083333
- **validation_R2_mean** (mean, AE phase, non-decisional): 0.690356
- **AE MLflow Run IDs:** 6f794e0e124c4644bea29b0fe44ac7f1 2b3af037b9804b6a87f993a09ba44a26 5fc589cf21cd40c889a5aeb7d65f64c9
- **Classification MLflow Run IDs:** 93366cd46fed4b7c958409ba92e8dba8 8ab081280011459b964a2f3fbeba6365 cf491c216de641f7ba034af389706137

## Training Dynamics
<!-- Agent, after the run: stability, convergence speed, spikes, plateau, early stopping. -->

## Conclusion
<!-- Agent, after the run: did the hypothesis hold? Mechanistic explanation of why it
     worked or failed — not just the numbers. -->
