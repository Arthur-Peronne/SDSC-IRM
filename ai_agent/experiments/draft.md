---
model_name: AE3dAsymResSeparableV2
summary: weight_decay 1e-5 -> 1e-3 (100x) on unchanged champion architecture, testing a whole-network regularization lever after 3 channel-width variations (trials 4-6) all failed
parent: 3aa0388f
id: null
status: draft
verdict: null
created_at: null
metric:
  primary: {name: classification_accuracy_val, value: null, direction: maximize}
---

# Trial <id> — AE3dAsymResSeparableV2 — <verdict>

<!-- ===== written BEFORE the run (agent) ===== -->

## Hypothesis
Trials 4-6 (b606a10f, 0cadad28, 001b5a08) tried three channel-width variations of the
champion architecture (widen-all, widen-encoder-only, halve-all) and all three failed to
beat classification_accuracy_val=0.6083, while validation_R2_mean stayed flat-to-better
in every case — evidence that channel width, in either direction, is not the lever that
improves this metric, and that reconstruction fidelity is not capacity-bottlenecked in
the range tested. Per program.md's stop-early guidance, I am not proposing a 4th
width variation; this trial changes a different regularization lever instead, with the
architecture held IDENTICAL to champion 3aa0388f. `weight_decay` (L2 penalty on ALL
network weights, currently 1e-5 — a very mild value, unchanged since trial 1's initial
default) is a fundamentally different mechanism from champion's bottleneck-only
dropout_rate=0.3: dropout regularizes which bottleneck units the classifier can rely on,
while weight_decay directly shrinks the magnitude of every weight in the encoder,
bottleneck, and decoder, discouraging the network from fitting high-magnitude,
train-specific weights anywhere in the pipeline. Given the campaign's recurring theme
(trials 4-6) that this architecture/data-volume combination is sensitive to
overfitting/variance regardless of capacity direction, I predict a substantially
stronger weight_decay (1e-3, 100x champion) reduces this overfitting further and
improves classification_accuracy_val above 0.6083, without needing to touch capacity at
all.

## Implementation
In `configs/autoencoder.yaml` (the only mutable file touched this trial): `weight_decay:
1e-5 -> 1e-3` (100x increase). Unchanged from champion 3aa0388f: `lr=5e-4`,
`dropout_rate=0.3`, `noise_std=0.0`, `patience=20`, `latent_dimensions=20`,
`model_name=AE3dAsymResSeparableV2`. No architecture file touched this trial — isolates
the weight_decay effect alone against the exact champion architecture (unaltered from
`src/models/ae_models.py`, reverted after trial 6's failure).

<!-- ===== written AFTER the run ===== -->

## Results
<!-- Filled automatically by the driver — leave empty. -->

## Training Dynamics
<!-- Agent, after the run: stability, convergence speed, spikes, plateau, early stopping. -->

## Conclusion
<!-- Agent, after the run: did the hypothesis hold? Mechanistic explanation of why it
     worked or failed — not just the numbers. -->
