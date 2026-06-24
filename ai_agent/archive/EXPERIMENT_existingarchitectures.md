# 🧪 PRELIMINARY EXPERIMENT: EXISTING ARCHITECTURES TEST

**Goal:** Validate the AI Agent's ability to follow the protocol by iterating through all existing architectures in `ae_models.py` (except the `AE3dFCDeep` baseline).

## 📊 BASELINE (Trial 0)
- **Run ID:** `4a0b3dd5cdae4727b1a966ce9e425268`
- **Model:** `AE3dFCDeep`
- **Latent Dim:** `120`
- **Metric (`val_mse`):** `0.000893567`

## 🔬 PROTOCOL FOR THIS TEST

### Phase 1: Preparation
1. **Identify Targets:** Identify all available architectures in `src/models/ae_models.py`. Cross-reference this list with `ai_agent/trial_log.csv` and exclude any models that have already been recorded in the log to avoid redundant trials.
2. **Reset State:** Ensure `ai_agent/trial_log.csv` is ready and `configs/autoencoder.yaml` is clean.

### Phase 2: Iterative Execution
For each target architecture:
1. **Configuration:** Modify `configs/autoencoder.yaml` to set the `model_name` to the current target 
and `experiment_tag: "aiagent_<model_name>"`
2. **Execution:** Run the training command with the specific test tag:
   `python -u scripts/run_autoencoder.py | tee training_baseline_aiagentref_<model_name>.log`
3. **Cleanup:** Immediately revert the configuration file to maintain a clean state for the next iteration:
   `git checkout HEAD -- configs/autoencoder.yaml`

### Phase 3: Logging & Committing
1. **Logging:** Log all metrics (`val_mse`, etc.) and perform a high-quality analysis in the `notes` column of `ai_agent/trial_log.csv`.

**Note Quality Standard:** Do NOT simply restate the numerical outcome. Notes must include:
- **Mechanism:** The theoretical reason for the performance (e.g., "Higher capacity via deeper layers").
- **Dynamics:** Training behavior (e.g., "Smooth convergence," "Instability observed").
2. **Simulated Success Commit:** To ensure all results are preserved, treat every trial as a **Success**:
    - **Include in commit:** `ai_agent/trial_log.csv`, `src/models/ae_models.py`, and `mlruns/`.
    - **Commit message format:** `AIagent baseline test: <model_name>`
