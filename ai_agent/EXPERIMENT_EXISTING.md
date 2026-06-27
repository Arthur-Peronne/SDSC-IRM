# 🧪 EXISTING ARCHITECTURES REFERENCE SWEEP

**Goal:** Establish `avg_validation_R2_mean` for all 9 existing architectures under the multi-dim protocol (latent_dims: 8, 60, 240). Results serve as reference baselines for `EXPERIMENT.md` and determine the starting champion under the new protocol.

## 📋 ARCHITECTURE QUEUE — update after each trial

| # | Model | Status | avg_R2_mean | mlflow_run_ids |
|---|---|---|---|---|
| 1 | `AE3dCurrent` | DONE | 0.713189 | efc10e1740eb4bed8ec77e88d5bb2757 2a409469cc5f4c84861db97f1405eeb6 a4e9410aeeda47f4b66f967fee036d5f |
| 2 | `AE3dFCDeep` | TODO | — | — |
| 3 | `AE3dConv` | TODO | — | — |
| 4 | `AE3dLinear` | TODO | — | — |
| 5 | `AE3dFCDeep_VAE` | TODO | — | — |
| 6 | `AE3dAttention` | TODO | — | — |
| 7 | `AE3dDilated` | TODO | — | — |
| 8 | `AE3dDilatedAttention` | TODO | — | — |
| 9 | `AE3dSeparableDilated` | TODO | — | — |

## 🔬 EXPERIMENTAL PHASES

### Phase 1: Select next model
**Re-read this file (`ai_agent/EXPERIMENT_EXISTING.md`) at the start of each iteration to get the current queue state.**

Find the first model with status `TODO` in the queue above. If all models are `DONE`, proceed to the **Post-Sweep** step below — the sweep is complete.

### Phase 2: Training
No new architecture implementation. Only configure and run.

1. **Preparation:** In `configs/autoencoder.yaml`, set `models_list: ["<ModelName>"]` and `experiment_tag: "aiagent_existing_<ModelName>"`. Confirm that `multiple_models_and_dims: true` and `latdim_list: [8, 60, 240]` are already set. These are the only fields to change.
2. **Execution:** Run training in the background (Bash tool `run_in_background=true`):
   `python -u scripts/run_autoencoder.py > results/0_tempo/training_aiagent_existing_<ModelName>.log 2>&1`
3. After the background process completes, read the log file to verify training finished without errors. If the log shows a crash, mark the model as `ERROR` in the queue, revert `configs/autoencoder.yaml`, and move to the next model.

### Phase 3: Log and commit
**Before anything else, re-read this file (`ai_agent/EXPERIMENT_EXISTING.md`) to ensure Phase 3 instructions are current in context.**

All trials are committed — there is no FAILURE/revert in this sweep.

1. **Extract metrics from MLflow** for all 3 runs. Replace `MODELNAME` accordingly:
   ```bash
   python3 -c "
   import mlflow
   mlflow.set_tracking_uri('mlruns')
   df = mlflow.search_runs(
       experiment_names=['autoencoder'],
       filter_string=\"params.experiment_tag = 'aiagent_existing_MODELNAME' and params.model_name = 'MODELNAME'\",
       order_by=['start_time DESC']
   )[['run_id', 'params.latent_dimensions', 'metrics.validation_R2_mean']].head(3)
   print(df.to_string(index=False))
   avg = df['metrics.validation_R2_mean'].mean()
   print(f'avg_R2_mean={avg:.6f}')
   "
   ```
   Verify that exactly 3 rows are returned. If fewer, check the training log for errors.

2. **Update the queue:** In the **📋 ARCHITECTURE QUEUE** table above, set the model's status to `DONE`, fill in `avg_R2_mean` and `mlflow_run_ids` (space-separated). Then add `ai_agent/EXPERIMENT_EXISTING.md` to the commit (step 5).

3. **Log results:** Append one row to `ai_agent/trials_conclusions/trial_log.csv`. Leave `trial_id`, `modification_description`, and `delta_vs_champion` blank; use `REFERENCE` as status:
   ```bash
   echo 'DATE,,MODEL,"8,60,240",,R2_8,R2_60,R2_240,AVG,,REFERENCE,"RUNID_8 RUNID_60 RUNID_240"' >> ai_agent/trials_conclusions/trial_log.csv
   ```
   Verify: `tail -1 ai_agent/trials_conclusions/trial_log.csv`

4. **Trial report:** Write `ai_agent/trials_conclusions/existing_<ModelName>.md`. Use `TEMPLATE.md` as a guide but replace the **Hypothesis** section with **Architecture Description** (a brief description of the model's design and how it differs from the others). Fill all other sections with mechanistic reasoning, not just numerical outcomes.

5. **Commit:**
   ```bash
   git checkout HEAD -- configs/autoencoder.yaml
   git add ai_agent/EXPERIMENT_EXISTING.md \
           ai_agent/trials_conclusions/trial_log.csv \
           ai_agent/trials_conclusions/existing_<ModelName>.md \
           mlruns/
   git commit -m "AIagent existing arch: <ModelName>"
   ```

## 🏁 Post-Sweep: Set the starting champion in EXPERIMENT.md
Once all 9 models are `DONE`, identify the model with the highest `avg_R2_mean` in the queue table. Then update the **🏆 CURRENT CHAMPION** block in `ai_agent/EXPERIMENT.md` with that model's metrics and run IDs, and commit `ai_agent/EXPERIMENT.md`.
