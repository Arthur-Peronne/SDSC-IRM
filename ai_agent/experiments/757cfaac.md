---
model_name: null
summary: null
parent: null
id: 757cfaac
status: completed
verdict: BASELINE
created_at: '2026-07-26T16:54:17+00:00'
metric:
  primary:
    name: classification_accuracy_val
    value: 0.458333
    direction: maximize
---

# Trial 757cfaac — ? — BASELINE

<!-- ===== written BEFORE the run (agent) ===== -->

## Hypothesis
<!-- What, why, how: "I modified X in Y because it will address Z via mechanism W,
     which I predict increases avg_validation_R2_mean."
     If this trial fuses two prior ideas, name both parents and describe the fusion. -->

## Implementation
<!-- The concrete architectural / hyperparameter change: what was added, where, and
     how it differs from the current champion. Respect the "no skip connections" rule. -->

<!-- ===== written AFTER the run ===== -->

## Results
- **accuracy_test per run:** 1b223966: 0.425000 | ab10db59: 0.550000 | b17955c4: 0.400000
- **classification_accuracy_val:** 0.458333
- **delta_vs_champion** (display only): +0.000000
- **validation_R2_mean** (mean, non-decisional): nan
- **AE MLflow Run IDs:** 1ab7d411b2d84a96a078a1f0e3524115 fb26900166364a4faf870f1a6d22a609 c354099975a143dab541fb44bcc3ec90
- **Classification MLflow Run IDs:** 1b223966be884fc6b1a80ada34d99c00 ab10db59eed54fea9a6269c1f64fdeb7 b17955c45e4f4f58a683863e4d4cbac2

## Training Dynamics
<!-- Agent, after the run: stability, convergence speed, spikes, plateau, early stopping. -->

## Conclusion
<!-- Agent, after the run: did the hypothesis hold? Mechanistic explanation of why it
     worked or failed — not just the numbers. -->