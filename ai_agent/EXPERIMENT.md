# 🧪 AE OPTIMIZATION EXPERIMENT PROTOCOL

## 📊 BASELINE (Trial 0) -> already done 
- **Run ID:** `4a0b3dd5cdae4727b1a966ce9e425268`
- **Model:** `AE3dFCDeep`
- **Latent Dim:** `120`
- **Metric (`val_mse`):** `0.000893567`

## 🔬 EXPERIMENTAL PHASES

### Phase 1: Hypothesis Generation
NB: no hyperparameter change, only AE architecture. No skip connections.
1. **Analyze:** Read previous trial results in `trial_log.csv`. Identify the "Current Champion" (best `val_mse`) and patterns in success/failure.
2. **Select a Strategy:**
    - **Exploration (New Architectures):** Propose a model from a new architectural family (e.g., Dilated, Multi-scale, Separable, or Topology changes).
    - **Exploitation (Refinement):** Propose an improvement to the Current Champion or a highly promising previous model (e.g., "Add Residual blocks to `AE3dAttention`").
3. **Formulate a Hypothesis:** State the *what*, *why*, and *how*: *"I will [Modify X] in [Model Y] because it will [Address Problem Z] via [Mechanism W], which I predict will decrease `val_mse`."*
4. **Design the Implementation:** Translate the hypothesis into a concrete architecture design, ensuring it respects the **"No Skip Connections"** rule.


### Phase 2: Controlled Implementation & Execution
1. **Implementation:** Implement the AE architecture in `ae_models.py` (and others if necessary).
2. **Preparation:** Name the trial and set `experiment_tag: "aiagent_NAME"` in `autoencoder.yaml`, and put the new model name in model_name.
3. **Execution:** `python -u scripts/run_autoencoder.py | tee training_<experiment_tag>.log`.

### Phase 3: Comparison and selection
1. **Log results:** Log model performances in trial_log.csv
2. **Metric comparison and decision** Compare `val_mse` and $\Delta \text{MSE} against the **best model to date** (not just the original baseline), and ensure predictive power is stable.
- **Success:** $\Delta \text{MSE} < 0$ AND stable loss curve 
- **Failure:** $\Delta \text{MSE} \geq 0$, unstable loss, or regression collapse
3. **Analysis** Document a scientific analysis in the `notes` column of `trial_log.csv`. 

**Note Quality Standard (CRITICAL):** Avoid tautologies (e.g., "Improved over baseline"). Every note MUST include:
1. **Mechanistic Reasoning:** The theoretical "Why" (e.g., "Spatial context preserved by avoiding flattening").
2. **Training Dynamics:** Observations on stability and convergence (e.g., "Stable loss curve," "Rapid convergence").
3. **Hypothesis Validation:** Whether the results align with or refute the initial hypothesis.
4. **Commit** Always add ai_agent/trial_log.csv and revert `configs/autoencoder.yaml` using `git checkout HEAD -- configs/autoencoder.yaml`. If Success, add src/models/ae_models.py and mlruns/. If Failure, revert src/models/ae_models.py and mlruns/ using `git checkout HEAD --`. Then commit -m "AIagent automatic MODEL_NAME".
