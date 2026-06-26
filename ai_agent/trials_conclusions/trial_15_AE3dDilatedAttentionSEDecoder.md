# Trial 15 — AE3dDilatedAttentionSEDecoder — FAILURE

## Hypothesis
I will add SE channel attention to all 4 decoder blocks of the champion (`AE3dDilatedAttention`), keeping the encoder and bottleneck identical. The champion's decoder uses plain `UpConv3DBlock` — no attention mechanism guides which channels to amplify during reconstruction. SE attention in the decoder lets the model selectively recalibrate feature channels at each upsampling step, potentially improving reconstruction of high-frequency details (edges, cardiac wall boundaries) that are hardest to reconstruct from a compressed latent representation. This is the first trial to target the decoder rather than the encoder.

## Implementation
New `AttentionUpConv3DBlock`: `UpConv3DBlock` with `SEBlock3D(reduction=4)` applied after the two conv layers. `reduction=4` used (instead of champion's 16) to avoid degenerate bottleneck at low channel counts (dec4 has 8 channels: 8//4=2). `AutoEncoder3D_DilatedAttentionSEDecoder` uses `AttentionUpConv3DBlock` for dec1–dec4; encoder (enc1–enc4 with dilations 1,2,4,1) and bottleneck are byte-for-byte identical to the champion.

## Results
- **validation_R2_mean:** 0.802384
- **validation_R2_std:** 0.076517
- **val_R2_lower_bound** (mean − std): 0.725868
- **lower_bound_compared_to_champion** (trial lb − champion lb): -0.004867
- **mean_compared_to_champion** (trial mean − champion mean): -0.001254
- **MLflow Run ID:** 7df080ba97ec4b839aacc72f2c46b48d
- **Best epoch:** 74 / 104 (early stop)

## Training Dynamics
Stable convergence, best epoch 74 — later than most previous trials (~46–50 epochs), suggesting the SE decoder adds meaningful learning signal that extends useful training. The model trained for 104 epochs before early stopping, the longest run since the champion (which ran to ~similar length). Validation std (0.077) closely matches the champion's (0.073), indicating the decoder SE does not introduce additional inter-patient variance.

## Conclusion
Statistically a FAILURE by protocol (both lb and mean marginally below champion), but architecturally the closest result since the champion itself — only 0.001 mean R2 and 0.005 lower bound below. This is the strongest candidate architecture found in trials 7–15.

The decoder SE hypothesis partially holds: adding attention to the decoder does not degrade performance (unlike most other modifications) and produces near-champion results. The SE decoder appears to improve reconstruction quality in a way that is genuinely complementary to the encoder SE — the stable std (0.077 vs champion 0.073) is the key signal that this modification does not add noise.

Why it falls just short: the decoder SE adds a small number of parameters that marginally increase the optimization difficulty. The champion's plain decoder may have a slight training efficiency advantage — fewer parameters to optimize, faster convergence to the global reconstruction optimum. The gap is small enough (Δlb=-0.005) that a different random initialization or longer patience could cross the threshold.

This architecture is worth revisiting in future hyperparameter optimization — it may outperform the champion with tuned patience, LR, or dropout.
