# 🧪 AE OPTIMIZATION EXPERIMENT PROTOCOL

## 📊 BASELINE (Trial 1) -> already done 
- **Run ID:** `4a0b3dd5cdae4727b1a966ce9e425268`
- **Model:** `AE3dFCDeep`
- **Latent Dim:** `120`
- **validation_R2_mean:** `0.688289` | **validation_R2_std:** `0.205251` | **val_R2_lower_bound:** `0.483038`

## 🏆 CURRENT CHAMPION — update this block on every CHAMPION
- **Trial:** 6
- **Model:** `AE3dDilatedAttention`
- **validation_R2_mean:** `0.803638` | **validation_R2_std:** `0.072903` | **val_R2_lower_bound:** `0.730735`
- **MLflow Run ID:** `1f8c81907c5d4fa8859606c0265b0a7c`

## 🔬 EXPERIMENTAL PHASES

### Phase 1: Hypothesis Generation
**Before anything else, re-read this file (`ai_agent/EXPERIMENT.md`) and `ai_agent/trials_conclusions/trial_log.csv` to ensure the current champion and trial history are fresh in context.**

NB: no hyperparameter change, only AE architecture. No skip connections.
1. **Analyze:** The Current Champion's metrics are in the **🏆 CURRENT CHAMPION** block above — no need to re-derive them from the CSV. Read `ai_agent/trials_conclusions/trial_log.csv` only to identify patterns across CHAMPION / CANDIDATE / FAILURE outcomes.
2. **Select a Strategy:**
    - **Exploration cooldown rule:** Count the last N consecutive FAILURE trials in `trial_log.csv`. If N ≥ 2 and all those failures are variants of the Current Champion (their model name starts with the champion's name), the next trial **must** be Exploration — proposing a model from a different architectural family. Exploitation is not permitted until a new champion is established.
    - **Exploration (New Architectures):** Propose a model from a new architectural family (e.g., Dilated, Multi-scale, Separable, or Topology changes). See the **Unexplored Directions** list below for concrete starting points — but that list is not exhaustive; propose anything architecturally sound that hasn't been tried.
    - **Exploitation (Refinement):** Tune the structure of the Current Champion or a highly promising previous model (e.g., change the number of layers, adjust channel widths, or modify kernel sizes).
3. **Formulate a Hypothesis:** State the *what*, *why*, and *how*: *"I will [Modify X] in [Model Y] because it will [Address Problem Z] via [Mechanism W], which I predict will increase `val_R2_lower_bound`."*
4. **Design the Implementation:** Translate the hypothesis into a concrete architecture design, ensuring it respects the **"No Skip Connections"** rule.

### 💡 Unexplored Directions
The following architectural ideas have **never been tried** in this project. This list is a starting point, not a constraint — if you have a well-motivated idea outside of it, propose it.

- **Strided convolution downsampling** — replace MaxPool with stride-2 convolutions; learned downsampling may preserve more task-relevant structure.
- **Spatial attention (CBAM-style)** — channel SE is tried; spatial attention (which spatial locations to focus on) has not been explored.
- **Multi-scale encoder** — parallel encoder paths at different dilations or resolutions, merged before the bottleneck.
- **Asymmetric depth** — lighter encoder, deeper decoder (or vice versa); current architecture is symmetric.
- **Different normalization** — GroupNorm or no normalization instead of InstanceNorm; may interact differently with small batch sizes.
- **1×1×1 bottleneck compression** — an explicit spatial collapse via 1×1×1 conv before the FC projection, instead of direct flattening.

### Phase 2: Controlled Implementation & Execution
1. **Implementation:** Implement the AE architecture in `src/models/ae_models.py`.
   - **Read first:** Before any edit, read the file to capture its current state.
   - **Insertion point:** Add the new class **immediately above** the `build_autoencoder` function (find it with `grep -n "def build_autoencoder" src/models/ae_models.py`). Never append at the end of the file.
   - **Factory registration:** Add exactly **one new `elif` branch** to `build_autoencoder` for the new model. Do not rewrite or reproduce any existing branch.
   - **No duplication:** Never reproduce existing class definitions. If an edit fails, re-read the file before retrying — do not re-apply the same edit blindly.
2. **Preparation:** In `configs/autoencoder.yaml`, set `model_name` to the new model's name and `experiment_tag: "aiagent_<NAME>"`. These are the only two fields to change.
3. **Execution:** `python -u scripts/run_autoencoder.py | tee results/0_tempo/training_<experiment_tag>.log`.

### Phase 3: Comparison and selection
**Before anything else, re-read this file (`ai_agent/EXPERIMENT.md`) to ensure Phase 3 instructions are current in context.**
1. **Extract metrics from MLflow** for the trained model's run ID:
   ```bash
   python3 -c "
   import mlflow
   mlflow.set_tracking_uri('mlruns')
   run = mlflow.get_run('RUN_ID')
   m = run.data.metrics
   mean = m['validation_R2_mean']
   std  = m['validation_R2_std']
   lb   = mean - std
   print(f'R2_mean={mean:.6f}  R2_std={std:.6f}  lb={lb:.6f}')
   "
   ```
2. **Decision** — read `val_R2_lower_bound` and `validation_R2_mean` from the **🏆 CURRENT CHAMPION** block above, then:
   - **CHAMPION:** `trial_lb > champion_lb` → new champion, update the block
   - **CANDIDATE:** `trial_lb ≤ champion_lb` AND `trial_mean > champion_mean` → save, do not update champion block
   - **FAILURE:** `trial_lb ≤ champion_lb` AND `trial_mean ≤ champion_mean` → revert
3. **Log results:** Append one row to `ai_agent/trials_conclusions/trial_log.csv` using the exact column order below:
   ```
   timestamp,trial_id,model_name,latent_dim,modification_description,validation_R2_mean,validation_R2_std,val_R2_lower_bound,lower_bound_compared_to_champion,mean_compared_to_champion,status,mlflow_run_id
   ```
   Append command (replace all-caps placeholders; `LB_VS_CHAMP` = trial_lb − champion_lb, `MEAN_VS_CHAMP` = trial_mean − champion_mean):
   ```bash
   echo "DATE,ID,MODEL,120,DESCRIPTION,R2_MEAN,R2_STD,LB,LB_VS_CHAMP,MEAN_VS_CHAMP,CHAMPION_OR_CANDIDATE_OR_FAILURE,RUN_ID" >> ai_agent/trials_conclusions/trial_log.csv
   ```
   Verify the row was appended correctly: `tail -1 ai_agent/trials_conclusions/trial_log.csv`
4. **Trial report:** Write `ai_agent/trials_conclusions/trial_<ID>_<ModelName>.md` using `TEMPLATE.md` in the same folder as a guide. Fill in all sections — do not leave placeholders. Avoid tautologies ("improved over baseline"). Every section must include mechanistic reasoning, not just numerical outcomes.
5. **Commit** — always add `ai_agent/trials_conclusions/trial_log.csv`, `ai_agent/trials_conclusions/trial_<ID>_<ModelName>.md`, and revert `configs/autoencoder.yaml` using `git checkout HEAD -- configs/autoencoder.yaml`. Then:
   - **CHAMPION:** update the **🏆 CURRENT CHAMPION** block at the top of this file, then add `src/models/ae_models.py`, `ai_agent/EXPERIMENT.md`, and `mlruns/`.
   - **CANDIDATE:** add `src/models/ae_models.py` and `mlruns/` (model and run are kept for future analysis — do **not** update the champion block).
   - **FAILURE:** revert `src/models/ae_models.py` using `git checkout HEAD -- src/models/ae_models.py` and purge the MLflow run using `git checkout HEAD -- mlruns/ && git clean -fd mlruns/`.
   
   Then commit: `git commit -m "AIagent automatic MODEL_NAME"`
