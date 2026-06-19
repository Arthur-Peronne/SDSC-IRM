# 🧪 AE OPTIMIZATION EXPERIMENT PROTOCOL

## 📊 BASELINE (Trial 0) -> already done 
- **Run ID:** `4a0b3dd5cdae4727b1a966ce9e425268`
- **Model:** `AE3dFCDeep`
- **Latent Dim:** `120`
- **Metric (`val_mse`):** `0.000893567`

## 🔬 EXPERIMENTAL PHASES

### Phase 1: New AE architecture hypothesis
NB: no hyperparamter change, only AE architecture. No skip connections.
1. **Check:** Read previous trial results in trial_log.csv 
2. **Idea:** Based on these results, the problem context and your imagination, find a new AE architecture idea
3. **Refining:** Refine your idea into a fully applicable AE architecture

### Phase 2: Controlled Implementation & Execution
1. **Implementation:** Implement the AE architecture in `ae_models.py` (and others if necessary).
2. **Preparation:** Name the trial and set `experiment_tag: "aiagent_NAME"` in `autoencoder.yaml`, and put the new model name in model_name.
3. **Execution:** `python -u scripts/run_autoencoder.py | tee training_<experiment_tag>.log`.

### Phase 3: Comparison and selection
1. **Log results:** Log model performances in trial_log.csv
2. **Metric comparison and decision** Compare `val_mse` and $\Delta \text{MSE} against the **best model to date** (not just the original baseline), and ensure predictive power is stable.
- **Success:** $\Delta \text{MSE} < 0$ AND stable loss curve 
- **Failure:** $\Delta \text{MSE} \geq 0$, unstable loss, or regression collapse
3. **Analysis** Analyse reasons of success or failure of this architecture, and save in notes in trial_log.csv
4. **Commit** Always add ai_agent/trial_log.csv and revert cOKonfigs/autoencoder.yaml. If Success, add src/models/ae_models.py and mlruns/. If Failure, revert src/models/ae_models.py and mlruns/. Then commit -m "AIagent automatic MODEL_NAME".
