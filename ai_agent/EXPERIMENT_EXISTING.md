# 🧪 EXISTING ARCHITECTURES REFERENCE SWEEP

**Goal:** Establish `avg_validation_R2_mean` for all 9 existing architectures under the multi-dim protocol (latent_dims: 8, 60, 240). Results serve as reference baselines for `EXPERIMENT.md` and determine the starting champion under the new protocol.

## 📋 ARCHITECTURE QUEUE — update after each trial

| # | Model | Status | avg_R2_mean | mlflow_run_ids |
|---|---|---|---|---|
| 1 | `AE3dCurrent` | DONE | 0.713189 | efc10e1740eb4bed8ec77e88d5bb2757 2a409469cc5f4c84861db97f1405eeb6 a4e9410aeeda47f4b66f967fee036d5f |
| 2 | `AE3dFCDeep` | DONE | 0.751709 | 0dee39dc65294cb69d33f1c99ea60ce0 9d1b0b8b751a4208911bc5ac714702a2 5140426136f643b499b089003010c28a |
| 3 | `AE3dConv` | DONE | 0.744460 | b2733e91203f45558850e66a7f89bab3 2db3eddbff364a348dbee128173bd591 17b3168b5bb3439f80bf976234d9bc33 |
| 4 | `AE3dLinear` | DONE | 0.590524 | d3cd4b195aee4d78943cd8eb4a897b38 fb34b6cd9d064492adafa7455e85c82e de3854f2406f485b9b32a4730bd4ba8f |
| 5 | `AE3dFCDeep_VAE` | DONE | 0.741339 | ab5b8c351c8c4fb5a8c554fafb284ab2 8449767e1e684d51a1e8e63967a6cf8a 3f1d8794d0994438a9b38ea6f4281d19 |
| 6 | `AE3dAttention` | DONE | 0.745631 | 65a449973c9b4819ae7c788328f7c7b8 b7d2a9522a6547cc9ef65cd9a252359c 40e9f1ef9f864a0b8a6696dbf3d57ad0 |
| 7 | `AE3dDilated` | DONE | 0.748160 | 0e338713893f49f09e6e9efbd595512f 72d1c817ae5d4cd8937f080828956f8e d5350a5061ed4c948f9ede2ff70c1c7c |
| 8 | `AE3dDilatedAttention` | DONE | 0.733706 | 2492702da6254a649b1c597ee382daf8 e6e3180bc5024dc2abcf8613db928641 3e03d6da5bc74c3bbac2c28a51a3d9ab |
| 9 | `AE3dSeparableDilated` | DONE | 0.737950 | d1e732e69dfd4d558c4a1f01aaed9d72 b31031c8e124440dbaf49d1ab961b042 f908acc9a5734addb7bac4b88a7be137 |

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
