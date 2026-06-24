# 🧪 AE OPTIMIZATION EXPERIMENT PROTOCOL

## 📊 BASELINE (Trial 0) -> already done 
- **Run ID:** `4a0b3dd5cdae4727b1a966ce9e425268`
- **Model:** `AE3dFCDeep`
- **Latent Dim:** `120`
- **Metric (`val_mse`):** `0.000835246`

## 🏆 CURRENT CHAMPION — update this block on every SUCCESS
- **Trial:** 6
- **Model:** `AE3dDilatedAttention`
- **Metric (`val_mse`):** `0.000573453`
- **MLflow Run ID:** `1f8c81907c5d4fa8859606c0265b0a7c`

## 🔬 EXPERIMENTAL PHASES

### Phase 1: Hypothesis Generation
**Before anything else, re-read this file (`ai_agent/EXPERIMENT.md`) and `ai_agent/trials_conclusions/trial_log.csv` to ensure the current champion and trial history are fresh in context.**

NB: no hyperparameter change, only AE architecture. No skip connections.
1. **Analyze:** Read previous trial results in `ai_agent/trials_conclusions/trial_log.csv`. Identify the "Current Champion" (best `val_mse`) and patterns in success/failure.
2. **Select a Strategy:**
    - **Exploration (New Architectures):** Propose a model from a new architectural family (e.g., Dilated, Multi-scale, Separable, or Topology changes).
    - **Exploitation (Refinement):** Tune the structure of the Current Champion or a highly promising previous model (e.g., change the number of layers, adjust channel widths, or modify kernel sizes).
3. **Formulate a Hypothesis:** State the *what*, *why*, and *how*: *"I will [Modify X] in [Model Y] because it will [Address Problem Z] via [Mechanism W], which I predict will decrease `val_mse`."*
4. **Design the Implementation:** Translate the hypothesis into a concrete architecture design, ensuring it respects the **"No Skip Connections"** rule.

### Phase 2: Controlled Implementation & Execution
1. **Implementation:** Implement the AE architecture in `src/models/ae_models.py`.
   - **Read first:** Before any edit, read the file to capture its current state.
   - **Insertion point:** Add the new class **immediately above** the `build_autoencoder` function (find it with `grep -n "def build_autoencoder" src/models/ae_models.py`). Never append at the end of the file.
   - **Factory registration:** Add exactly **one new `elif` branch** to `build_autoencoder` for the new model. Do not rewrite or reproduce any existing branch.
   - **No duplication:** Never reproduce existing class definitions. If an edit fails, re-read the file before retrying — do not re-apply the same edit blindly.
2. **Preparation:** In `configs/autoencoder.yaml`, set `model_name` to the new model's name and `experiment_tag: "aiagent_<NAME>"`. These are the only two fields to change.
3. **Execution:** `python -u scripts/run_autoencoder.py | tee SDSC-IRM/training_<experiment_tag>.log`.

### Phase 3: Comparison and selection
**Before anything else, re-read this file (`ai_agent/EXPERIMENT.md`) to ensure Phase 3 instructions are current in context.**
1. **Log results:** Append one row to `ai_agent/trials_conclusions/trial_log.csv` using the exact column order below:
   ```
   timestamp,trial_id,model_name,latent_dim,modification_description,metric_name,metric_value,metric_delta,status,mlflow_run_id
   ```
   Append command (replace all-caps placeholders):
   ```bash
   echo "DATE,ID,MODEL,120,DESCRIPTION,val_mse,VALUE,DELTA,SUCCESS_OR_FAILURE,RUN_ID" >> ai_agent/trials_conclusions/trial_log.csv
   ```
   Verify the row was appended correctly: `tail -1 ai_agent/trials_conclusions/trial_log.csv`
2. **Metric comparison and decision** Compare `val_mse` and $\Delta \text{MSE}$ against the **best model to date** (not just the original baseline), and ensure predictive power is stable.
   - **Success:** $\Delta \text{MSE} < 0$ AND stable loss curve
   - **Failure:** $\Delta \text{MSE} \geq 0$, unstable loss, or regression collapse
3. **Trial report:** Write `ai_agent/trials_conclusions/trial_<ID>_<ModelName>.md` using `TEMPLATE.md` in the same folder as a guide. Fill in all sections — do not leave placeholders. Avoid tautologies ("improved over baseline"). Every section must include mechanistic reasoning, not just numerical outcomes.
4. **Commit** Always add `ai_agent/trials_conclusions/trial_log.csv` and `ai_agent/trials_conclusions/trial_<ID>_<ModelName>.md`, and revert `configs/autoencoder.yaml` using `git checkout HEAD -- configs/autoencoder.yaml`. If Success, update the **🏆 CURRENT CHAMPION** block at the top of this file with the new trial number, model name, val_mse, and MLflow run ID — then add `src/models/ae_models.py`, `ai_agent/EXPERIMENT.md`, and `mlruns/`. If Failure, revert `src/models/ae_models.py` and `mlruns/` using `git checkout HEAD --`. Then commit -m "AIagent automatic MODEL_NAME".

