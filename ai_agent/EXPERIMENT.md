# 🧪 AE OPTIMIZATION EXPERIMENT PROTOCOL

## 📊 BASELINE (historical — single latent_dim=120, old protocol)
- **Run ID:** `4a0b3dd5cdae4727b1a966ce9e425268`
- **Model:** `AE3dFCDeep`
- **validation_R2_mean:** `0.688289`

## 🏆 CURRENT CHAMPION — update this block on every CHAMPION
- **Trial:** — (none yet under multi-dim protocol)
- **Model:** —
- **avg_validation_R2_mean (latent_dims: 8, 60, 240):** —
- **MLflow Run IDs:** —
- **Note:** If champion_avg is —, the first trial automatically becomes CHAMPION.

## 🔬 EXPERIMENTAL PHASES

### Phase 1: Hypothesis Generation
**Before anything else, re-read this file (`ai_agent/EXPERIMENT.md`) and `ai_agent/trials_conclusions/trial_log.csv` to ensure the current champion and trial history are fresh in context.**

NB: no hyperparameter change, only AE architecture. No skip connections.
1. **Analyze:** The Current Champion's metrics are in the **🏆 CURRENT CHAMPION** block above — no need to re-derive them from the CSV. Read `ai_agent/trials_conclusions/trial_log.csv` only to identify patterns across CHAMPION / CANDIDATE / FAILURE outcomes.
2. **Select a Strategy:**
    - **Exploration cooldown rule:** Count the last N consecutive FAILURE trials in `trial_log.csv`. If N ≥ 2 and all those failures are variants of the Current Champion (their model name starts with the champion's name), the next trial **must** be Exploration — proposing a model from a different architectural family. Exploitation is not permitted until a new champion is established.
    - **Exploration (New Architectures):** Propose a model from a new architectural family (e.g., Dilated, Multi-scale, Separable, or Topology changes). See the **Unexplored Directions** list below for concrete starting points — but that list is not exhaustive; propose anything architecturally sound that hasn't been tried.
    - **Exploitation (Refinement):** Tune the structure of the Current Champion or a highly promising previous model (e.g., change the number of layers, adjust channel widths, or modify kernel sizes).
3. **Formulate a Hypothesis:** State the *what*, *why*, and *how*: *"I will [Modify X] in [Model Y] because it will [Address Problem Z] via [Mechanism W], which I predict will increase `avg_validation_R2_mean`."*
4. **Design the Implementation:** Translate the hypothesis into a concrete architecture design, ensuring it respects the **"No Skip Connections"** rule.

### 💡 Strategy Reference
Two valid sources for the next trial:
- **New ideas:** Propose any architecturally sound model not yet tried under the multi-dim protocol.
- **Archive retry:** Read `ai_agent/archive/experiment_architecture_1/trial_log.csv` and identify models with high `validation_R2_mean` (former CANDIDATEs or near-misses). These are worth re-running under the new protocol — a different latent dim distribution may yield a different ranking, and the old single-dim results were potentially noisy.

### Phase 2: Controlled Implementation & Execution
1. **Implementation:** Implement the AE architecture in `src/models/ae_models.py`.
   - **Read first:** Before any edit, read the file to capture its current state.
   - **Insertion point:** Add the new class **immediately above** the `build_autoencoder` function (find it with `grep -n "def build_autoencoder" src/models/ae_models.py`). Never append at the end of the file.
   - **Factory registration:** Add exactly **one new `elif` branch** to `build_autoencoder` for the new model. Do not rewrite or reproduce any existing branch.
   - **No duplication:** Never reproduce existing class definitions. If an edit fails, re-read the file before retrying — do not re-apply the same edit blindly.
2. **Preparation:** In `configs/autoencoder.yaml`, set `models_list: ["<ModelName>"]` and `experiment_tag: "aiagent_<NAME>"`. Confirm that `multiple_models_and_dims: true` and `latdim_list: [8, 60, 240]` are already set. These are the only fields to change.
3. **Execution:** Run training in the background (Bash tool `run_in_background=true`) with the command:
   `python -u scripts/run_autoencoder.py > results/0_tempo/training_<experiment_tag>.log 2>&1`
   After the background process completes, read the log file to verify training finished without errors.

### Phase 3: Comparison and selection
**Before anything else, re-read this file (`ai_agent/EXPERIMENT.md`) to ensure Phase 3 instructions are current in context.**
1. **Extract metrics from MLflow** for all 3 runs. Replace `EXPERIMENT_TAG` and `MODELNAME` with the actual values (e.g. `aiagent_AE3dFoo` and `AE3dFoo`):
   ```bash
   python3 -c "
   import mlflow
   mlflow.set_tracking_uri('mlruns')
   df = mlflow.search_runs(
       experiment_names=['autoencoder'],
       filter_string=\"params.experiment_tag = 'EXPERIMENT_TAG' and params.model_name = 'MODELNAME'\",
       order_by=['start_time DESC']
   )[['run_id', 'params.latent_dimensions', 'metrics.validation_R2_mean']].head(3)
   print(df.to_string(index=False))
   avg = df['metrics.validation_R2_mean'].mean()
   print(f'avg_R2_mean={avg:.6f}')
   "
   ```
   Verify that exactly 3 rows are returned (one per latent dim). If fewer, check the training log for errors.
2. **Decision** — read `avg_validation_R2_mean` (= `champion_avg`) from the **🏆 CURRENT CHAMPION** block above, then:
   - **CHAMPION:** `trial_avg > champion_avg` → new champion, update the block
   - **CANDIDATE:** `trial_avg ≤ champion_avg` AND `max(R2_dim8, R2_dim60, R2_dim240) > champion_avg + 0.03` → save, do not update champion block
   - **FAILURE:** neither CHAMPION nor CANDIDATE → revert
   - **No champion yet (champion_avg = —):** trial automatically becomes CHAMPION.
3. **Log results:** Append one row to `ai_agent/trials_conclusions/trial_log.csv` using the exact column order below:
   ```
   timestamp,trial_id,model_name,latent_dims,modification_description,R2_dim8,R2_dim60,R2_dim240,avg_validation_R2_mean,delta_vs_champion,status,mlflow_run_ids
   ```
   Append command (replace all-caps placeholders; `DELTA` = trial_avg − champion_avg; `RUN_IDS` = space-separated run IDs for dims 8/60/240):
   ```bash
   echo 'DATE,ID,MODEL,"8,60,240",DESCRIPTION,R2_8,R2_60,R2_240,AVG,DELTA,STATUS,"RUNID_8 RUNID_60 RUNID_240"' >> ai_agent/trials_conclusions/trial_log.csv
   ```
   Verify the row was appended correctly: `tail -1 ai_agent/trials_conclusions/trial_log.csv`
4. **Trial report:** Write `ai_agent/trials_conclusions/trial_<ID>_<ModelName>.md` using `TEMPLATE.md` in the same folder as a guide. Fill in all sections — do not leave placeholders. Avoid tautologies ("improved over baseline"). Every section must include mechanistic reasoning, not just numerical outcomes.
5. **Commit** — always add `ai_agent/trials_conclusions/trial_log.csv`, `ai_agent/trials_conclusions/trial_<ID>_<ModelName>.md`, and revert `configs/autoencoder.yaml` using `git checkout HEAD -- configs/autoencoder.yaml`. Then:
   - **CHAMPION:** update the **🏆 CURRENT CHAMPION** block at the top of this file, then add `src/models/ae_models.py`, `ai_agent/EXPERIMENT.md`, and `mlruns/`.
   - **CANDIDATE:** add `src/models/ae_models.py` and `mlruns/` (model and runs are kept for future analysis — do **not** update the champion block).
   - **FAILURE:** revert `src/models/ae_models.py` using `git checkout HEAD -- src/models/ae_models.py` and purge the MLflow run using `git checkout HEAD -- mlruns/ && git clean -fd mlruns/`.
   
   Then commit: `git commit -m "AIagent automatic MODEL_NAME"`
