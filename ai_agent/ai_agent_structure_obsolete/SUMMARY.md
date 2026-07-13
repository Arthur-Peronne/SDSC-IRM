# 📜 PROJECT OPERATIONAL SUMMARY: SDSC-IRM AE OPTIMIZATION

> **Read this file only if you need more historical context on the project.** For current rules, constraints, and workflow, use `AGENT.md` and `EXPERIMENT.md` instead. This file is structured chronologically and documents decisions, reasoning, and outcomes of past sessions.

## 🎯 1. MISSION OVERVIEW
The objective is to develop a 3D Autoencoder (AE) architecture for dimensionality reduction of IRM cardiac MRI images $(1, 32, 128, 128)$ that outperforms PCA in reconstruction fidelity ($R^2$) while maintaining high predictive power for downstream regression tasks.

### ⚖️ The Fundamental Trade-off
Current models exhibit high predictive power but suffer from **"MSE-induced blurriness."** They capture the "essence" (low-frequency structure) but fail to capture the "details" (high-frequency edges/textures). Our goal is to move the needle on reconstruction $R^2$ without sacrificing the latent space's predictive utility.

---

## 🏗️ 2. CORE ARCHITECTURAL CONSTRAINTS
To ensure the research remains valid for future stages (e.g., studying heart tissue deformations), we must adhere to a strict structural rule:

*   **STRICTLY NO SKIP CONNECTIONS (No U-Net):** The architecture **must not** allow spatial information to bypass the bottleneck via skip connections. 
*   **Rationale:** We need an "independent" latent space. Any information used for reconstruction must be compressed through the bottleneck. This ensures that any future constraints (like VAE-style regularization or disentanglement) act on the *entire* representation of the image, not just the residual details.

### 🛠️ Optimization Levers
1.  **Primary Metric:** `val_mse` (Validation Reconstruction Loss).
2.  **Secondary Metric:** Regression scores (Predictive Power).
3.  **Structural Directions:**
    *   **Residual 3D AE:** Implementing $x + \text{Conv}(x)$ blocks to enable deeper, more stable feature extraction.
    *   **Attention-Guided AE:** Integrating Squeeze-and-Excitation (SE) blocks to prioritize cardiac structures and suppress background noise.
    *   **Dilated 3D AE:** Using dilated convolutions to increase the receptive field (global context) without losing spatial resolution.

---

## ⚙️ 3. PHASE 1: ARCHITECTURAL REFACTORING (THE "KARPATHY" PRE-REQUISITE)
The current codebase is "hard-coded," meaning the training script must be manually edited to introduce new architecture-specific hyperparameters. This prevents automated, high-throughput experimentation.

### 💎 The Goal: Model-Agnostic Training Engine
We must decouple the **Training Logic** (the engine) from the **Model Architecture** (the components) by implementing a `**kwargs` based parameter passing system.

### 📝 Detailed Procedure
1.  **The Model Factory (`src/models/ae_models.py`):** 
    *   Modify the `build_autoencoder` function to accept `**kwargs`.
    *   Ensure all model classes (e.g., `AE3dCurrent`, `AE3dFCDeep`) are updated to accept and utilize these arguments in their `__init__` methods.
2.  **The Bridge (`scripts/run_autoencoder.py`):**
    *   Update the training script to load the YAML configuration into a dictionary.
    *   Separate "Meta-parameters" (e.g., `model_name`, `learning_rate`, `batch_size`) from "Architecture-parameters" (e.g., `num_layers`, `dilation_rate`, `dropout`).
    *   Pass the "Architecture-parameters" dictionary into `build_autoencoder` using the `**kwargs` unpacking operator.
3.  **The Verification:**
    *   Run baseline models (`AE3dCurrent`, `AE3dLinear`) to ensure the refactor has not introduced regressions.

### ⚠️ Risks & Mitigations
| Risk | Impact | Mitigation Strategy |
| :--- | :--- | :--- |
| **Argument Mismatch** | Training crashes if a parameter is passed that a model doesn't expect. | Use `kwargs.pop('param', default)` within model classes to handle unexpected or optional arguments gracefully. |
| **The "Black Hole" Problem** | Parameters are passed in the YAML but silently ignored by the model (no error, but no effect). | Implement rigorous logging that prints the *actual* parameters received by the model instance at initialization. |
| **Config Bloat** | The `autoencoder.yaml` becomes an unreadable mess of parameters for every possible model. | Maintain a clear distinction in the YAML structure between "Global Training Params" and "Model-Specific Params." |
| **Breaking Changes** | Refactoring core scripts might break existing MLflow/experiment tracking. | Perform full regression testing on all existing model types before declaring Phase 1 complete. |

---

## 🔄 4. OPERATIONAL WORKFLOW (AGENT LOOP)
1.  **Analyze:** Review `trial_log.csv` and current `val_mse`.
2.  **Propose:** Select a structural direction (Residual, Attention, or Dilated).
3.  **Modify:** Implement changes (either in `.yaml` or `.py`).
4.  **Execute:** `python -u scripts/run_autoencoder.py | tee training_<tag>.log`.
5.  **Evaluate:** 
    *   If `val_mse` improves $\rightarrow$ **COMMIT** and log to `trial_log.csv`.
    *   If `val_mse` fails $\rightarrow$ **REVERT** and try a different lever.

---

## 🧪 PROPOSED EXPERIMENT: ATTENTION-GUIDED AE

### 💡 Concept
Testing an **Attention-Guided 3D Autoencoder (`AE3dAttention`)** to combat "MSE-induced blurriness."

### 🛠️ Implementation Details
*   **Mechanism:** Integrate **Squeeze-and-Excitation (SE) blocks** into the convolutional blocks. 
*   **Goal:** Perform channel-wise feature recalibration to prioritize cardiac structures and suppress background noise.
*   **Comparison:** The new architecture will be a structural sibling to `AE3dCurrent` (same depth, same number of channels, same latent dimension) to ensure a fair, hyperparameter-neutral comparison.

### ✅ Compliance Check
*   **No Skip Connections:** SE blocks do not introduce spatial skip connections; the bottleneck remains the sole information conduit.
*   **No Hyperparameter Changes:** All other parameters will be kept identical to the baseline.

## 🚀 CURRENT STATUS: ATTENTION-GUIDED AE IMPLEMENTED
The `AE3dAttention` architecture has been implemented in `src/models/ae_models.py`. 
**Next Step:** Run baseline tests and start the optimization loop for the attention architecture.

---

## 📝 RECENT CONVERSATION SUMMARY
The agent encountered a loop during the previous session and was restarted. The following key points were identified and confirmed:
- **Loop-Breaking Protocol:** Confirmed awareness of the Two-Strike Rule, Mandatory Re-Reads, Granularity Shifts, and Strategy Pivots to prevent infinite retry loops.
- **Architectural Constraints:** Re-confirmed the "No Skip Connections" rule to maintain the independence of the latent space.
- **Current Implementation Status:** The `AE3dAttention` architecture is implemented in `src/models/ae_models.py` and integrated into the `build_autoencoder` factory function.
- **Immediate Objective:** Verify the implementation via baseline testing and begin the optimization loop for the attention-guided architecture using `val_mse` as the primary metric.

---

## 📅 SESSION LOG — 2026-06-19

### 1. trial_log.csv metric correction
- **Problem:** The CSV was logging `validation_R2_mean` as the primary metric, but the protocol specifies `val_mse`.
- **Action:** Restructured `trial_log.csv` with correct columns: `model_name`, `latent_dim`, `metric_name` (val_mse), `metric_value`, `metric_delta`, `mlflow_run_id`, `notes`.
- **Baseline value:** Pulled directly from MLflow (`4a0b3dd5cdae4727b1a966ce9e425268`): `val_mse = 0.0008935671168728732`.
- **EXPERIMENT.md updated** to match the full-precision baseline value.

### 2. Trial 1 result: AE3dAttention
- **Last model trained before crash:** AE3dAttention (2-epoch dryrun, catastrophic failure — R² = -36).
- **Full training result:** AE3dAttention with tag `aiagent_attention_1` (300 epochs, MLflow run `dccb003a0a5c4b779a19f17067c319a6`).
- **val_mse:** `0.0007721571491856594` (Δ = **-0.000121** vs baseline).
- **test_R2:** 0.750 → 0.774. **validation_R2:** 0.688 → 0.742.
- **Loss curve:** Stable, no spikes, no regression collapse. Best epoch 63 (vs 76 baseline). Early stopping at 93 (vs 106).
- **Analysis (logged in trial_log.csv):** SE blocks enabled channel-wise feature recalibration, prioritizing cardiac structures over background noise.

### 3. Phase 3 protocol refinement
- **Key decision:** ΔMSE is calculated against the **best model to date**, not the original baseline. A trial only commits if it beats the current best.
- **Phase 3 finalized as 4 steps:**
  1. Log results → trial_log.csv
  2. Compare ΔMSE vs best model to date (stable loss curve required)
  3. Analysis → document reasons in trial_log.csv notes
  4. Commit → always add trial_log.csv + revert configs/autoencoder.yaml. Success: add ae_models.py + mlruns/. Failure: revert ae_models.py + mlruns/. Then commit.
- **EXPERIMENT.md updated** with this refined protocol.

### 4. Commit
- **Commit `5e181a77`:** "AIagent automatic AE3dAttention"
- **Included:** `src/models/ae_models.py` (AE3dAttention code), `ai_agent/trial_log.csv`, `mlruns/` (already tracked)
- **Reverted:** `configs/autoencoder.yaml` back to `vae_optuna2` / `AE3dFCDeep_VAE`
- **Working tree:** clean, ready for next trial.

### 5. Documentation updates
- **CODEBASE.md:** Added note that other AI-Agent-created architectures exist but are not listed to avoid constant updates. Only the original 5 models are in the table.
- **AGENT.md:** Marked architecture test items as DONE in the to-do list.
- **SUMMARY.md:** This file — structured chronologically to document session decisions and their reasoning.

### 6. Critical lesson from this session
- **Never assume uncommitted changes are committed.** When reverting, always check `git diff --stat` and `git show HEAD` to understand what's actually in the repo vs what's in the working directory. The previous session had uncommitted changes that were lost when `git checkout HEAD` was used.


---


---

## 📅 SESSION LOG — 2026-06-20

### 1. Trial 4 result: AE3dDilated
- **Last model trained:** AE3dDilated (300 epochs, early stopping at 70).
- **val_mse:** `0.000642` (Δ = **-0.000130** vs champion `AE3dAttention`).
- **test_R2:** 0.7885.
- **Analysis (logged in trial_log.csv):** Dilated convolutions successfully increased the receptive field of the encoder layers, allowing the model to capture larger-scale cardiac structures without the need for pooling-induced resolution loss or skip connections. This directly addresses the "MSE-induced blurriness" by providing better global context to the bottleneck.
- **Status:** **SUCCESS**. New architecture champion.
- **Action:** Commit `ae_models.py` and `trial_log.csv`. Revert `autoencoder.yaml`.

---

## 📅 SESSION LOG — 2026-06-24: Documentation overhaul

This session focused entirely on improving the `ai_agent/` documentation to increase agent stability, based on failure patterns observed with previous models (Qwen, Gemma): inability to write results, falling into retry loops, and code duplication in `ae_models.py`.

### 1. Root causes identified
- **Technical Environment section was empty** (`AGENT.md`) — agents didn't know the working directory, Python executable, or key file paths.
- **No safe-edit protocol for `ae_models.py`** — no instruction to read before editing, no explicit insertion point, no warning against reproducing existing classes. Led to duplicate class definitions.
- **No explicit CSV append instruction** — agents inferred how to write results and sometimes overwrote the file or failed silently.
- **Inconsistent log file paths** across `AGENT.md` and `EXPERIMENT.md`.
- **Context overflow at Phase 3** — 300-epoch training output fills the agent's context, causing it to forget Phase 3 instructions read earlier in the session.
- **Baseline value mismatch** between `EXPERIMENT.md` (0.000893567) and `trial_log.csv` (0.000835246).

### 2. Key fixes applied

**AGENT.md:**
- Filled in Technical Environment: working directory, Python executable, git branch, key file paths, env verification command.
- Rephrased "MOST IMPORTANT RULE" to clearly scope file modification to files authorized by `EXPERIMENT.md`.
- Added a General Rule to the Loop-Breaking Protocol: stop and ask the user after two identical failures, not just on edit failures.
- Removed redundant operational details (already in `EXPERIMENT.md`), vague rules, done to-do items, and KARPATHY future work.

**EXPERIMENT.md:**
- Added safe-edit protocol to Phase 2: read before editing, insertion point above `build_autoencoder`, one `elif` only, no duplication.
- Added explicit CSV append command with column order and `tail -1` verification to Phase 3.
- Added re-read instruction at the start of Phase 3 to counter context overflow.
- Added `🏆 CURRENT CHAMPION` block at the top — updated on every SUCCESS, so the agent never needs to parse the CSV to find the reference value.
- Fixed baseline value and unified log file path to `SDSC-IRM/training_<tag>.log`.
- Removed redundant LOGGING CONVENTION block.

**trial_log.csv:**
- Removed `notes` column — qualitative analysis now goes in per-trial markdown reports.

**New: `ai_agent/trials_conclusions/`:**
- Contains `trial_log.csv` (quantitative results) and one `trial_<ID>_<ModelName>.md` per trial (qualitative report).
- `TEMPLATE.md` provides a fixed structure the agent fills in — prevents open-ended generation at the most context-heavy moment.

**CODEBASE.md:** Removed redundant project overview (covered by `AGENT.md`), renumbered sections.

**SUMMARY.md:** Added note directing agents to read this file only for historical context, not as part of the standard workflow.

### 3. Design decisions and rationale
- **Reassurance language rejected:** Phrases like "you're doing well, keep going" don't fix weak model failures; clearer step-by-step instructions do.
- **Skills (`/ae-phase3`) rejected:** A re-read instruction in the doc achieves the same context refresh without adding moving parts.
- **One report per trial, not cumulative:** Appending to a growing file risks overwriting previous entries; creating a new file per trial is a simpler, isolated write operation.
- **`notes` column removed from CSV:** Redundant with the markdown report; keeping both creates inconsistency risk.
- **`SUMMARY.md` kept (not archived):** Added a conditional-read note instead — useful for deep context, not needed for routine trials.

---

## 📅 SESSION LOG — 2026-06-25: Metric overhaul and protocol refinements

This session overhauled the decision metric and tightened the protocol. No new training was run.

### 1. Primary metric changed: val_mse → validation_R2 lower bound

`val_mse` was an online training metric (noisy, per-epoch). It has been replaced with post-training metrics computed on the restored best model:

- **`validation_R2_mean`** and **`validation_R2_std`** (across the 20 validation patients, logged by MLflow)
- **`val_R2_lower_bound`** = `mean − std` — the ranking metric

**Decision logic (replaces old Δval_mse < 0):**
- **CHAMPION:** `trial_lb > champion_lb`
- **CANDIDATE:** `trial_lb ≤ champion_lb` AND `trial_mean > champion_mean` — higher average but more variable; kept for future analysis
- **FAILURE:** `trial_lb ≤ champion_lb` AND `trial_mean ≤ champion_mean` — revert as before

The champion's `validation_R2_mean` and `val_R2_lower_bound` are pre-computed in the **🏆 CURRENT CHAMPION** block — the agent must read them from there, not re-derive from the CSV.

### 2. trial_log.csv columns updated

Old columns `metric_name`, `metric_value`, `metric_delta` replaced with:
`validation_R2_mean, validation_R2_std, val_R2_lower_bound, lower_bound_compared_to_champion, mean_compared_to_champion`

All 11 past trials backfilled with R2 data from MLflow (trials 1–7 full, trials 8–11 R2_mean only — std lost when mlruns were purged on failure). Statuses retroactively updated: trials 3–5, 7–8, 10–11 reclassified (were FAILURE under val_mse, now correctly classified under R2).

### 3. Phase 3 MLflow extraction command added

Agents now extract R2 metrics directly from MLflow at the start of Phase 3:
```bash
python3 -c "
import mlflow; mlflow.set_tracking_uri('mlruns')
run = mlflow.get_run('RUN_ID')
m = run.data.metrics
mean = m['validation_R2_mean']; std = m['validation_R2_std']; lb = mean - std
print(f'R2_mean={mean:.6f}  R2_std={std:.6f}  lb={lb:.6f}')
"
```

### 4. Exploration cooldown rule added (EXPERIMENT.md Phase 1)

**Rule:** If the last 2+ consecutive FAILURE trials are all variants of the Current Champion (model name starts with champion's name), the next trial **must** be Exploration. Exploitation is blocked until a new champion is established.

**Current state:** Trials 8–11 are 4 consecutive FAILUREs on AE3dDilatedAttention variants → **cooldown is already triggered. Trial 12 must be Exploration.**

### 5. Unexplored Directions section added (EXPERIMENT.md)

A curated list of architectural families never tried, placed between Phase 1 and Phase 2 for agent reference during mandatory exploration:
- Strided conv downsampling (instead of MaxPool)
- CBAM-style spatial attention (channel SE already tried)
- Multi-scale parallel encoder paths
- Asymmetric encoder-decoder depth
- GroupNorm or no normalization
- 1×1×1 bottleneck compression before FC

The list is explicitly non-exhaustive — agents may propose anything architecturally sound.

### 6. Minor protocol fixes

- **Log file path fixed:** `results/0_tempo/training_<tag>.log` (was `SDSC-IRM/training_<tag>.log`, wrong relative path)
- **MLflow artifact:** `scripts/run_autoencoder.py` now uploads the training log as an MLflow artifact at end of run (guarded by `if log_path.exists()`)
- **Champion block is authoritative:** Phase 1 step 1 no longer asks the agent to re-derive the champion from the CSV
- **Cooldown rule wording:** "model name starts with the champion's name" — unambiguous for string matching

---

## 📅 SESSION LOG — 2026-06-26: Multi-dim protocol overhaul + reference sweep setup

No new training was run. Full protocol redesign motivated by the observation that 28 trials in experiment_architecture_1 all failed to beat `AE3dDilatedAttention` — suspected cause: the champion's unusually low std=0.073 at latdim=120 may have been a lucky single run rather than a genuine architectural advantage.

### 1. Root causes identified

- **Single latent dim (120) per trial:** one training run at one dimension is highly noisy. A lucky convergence can produce an artificially strong champion that blocks all subsequent trials.
- **val_R2_lower_bound as metric:** `mean − std` amplifies variance — both terms fluctuate across runs, making the lb doubly noisy.
- **Champion benchmark invalid under new protocol:** `AE3dDilatedAttention`'s metrics were measured only at latdim=120; no comparable multi-dim baseline existed.

### 2. Multi-dim training adopted

Each trial now trains at **three latent dimensions: [8, 60, 240]**, producing 3 independent MLflow runs per trial. The `autoencoder.yaml` was updated: `multiple_models_and_dims: true`, `models_list: ["<ModelName>"]`, `latdim_list: [8, 60, 240]`.

**This yaml change must be committed before the loop starts** — each trial ends with `git checkout HEAD -- configs/autoencoder.yaml`, so the committed state is the revert target.

### 3. Decision metric changed: val_R2_lower_bound → avg_validation_R2_mean

The new ranking metric is the **average `validation_R2_mean` across the 3 latent dims**. Lower bound is dropped entirely.

New decision logic:
- **CHAMPION:** `trial_avg > champion_avg`
- **CANDIDATE:** `trial_avg > champion_avg − 0.03`
- **FAILURE:** `trial_avg ≤ champion_avg − 0.03`
- **No champion yet (—):** first trial automatically becomes CHAMPION

### 4. trial_log.csv columns updated

Old: `validation_R2_mean, validation_R2_std, val_R2_lower_bound, lower_bound_compared_to_champion, mean_compared_to_champion`

New: `latent_dims, R2_dim8, R2_dim60, R2_dim240, avg_validation_R2_mean, delta_vs_champion`

A fresh `trial_log.csv` was created with the new header. All 28 past trials are archived in `archive/experiment_architecture_1/`.

### 5. MLflow extraction updated

Phase 3 now searches by `experiment_tag + model_name` to retrieve all 3 runs, then computes the average:
```bash
python3 -c "
import mlflow; mlflow.set_tracking_uri('mlruns')
df = mlflow.search_runs(
    experiment_names=['autoencoder'],
    filter_string=\"params.experiment_tag = 'EXPERIMENT_TAG' and params.model_name = 'MODELNAME'\",
    order_by=['start_time DESC']
)[['run_id', 'params.latent_dimensions', 'metrics.validation_R2_mean']].head(3)
print(df.to_string(index=False)); print(f'avg={df[\"metrics.validation_R2_mean\"].mean():.6f}')
"
```

### 6. EXPERIMENT_EXISTING.md created — reference sweep of 9 architectures

Before running new architecture trials, all 9 registered architectures must be evaluated under the new multi-dim protocol to establish valid baselines and set the starting champion. The 9 models: `AE3dCurrent`, `AE3dFCDeep`, `AE3dConv`, `AE3dLinear`, `AE3dFCDeep_VAE`, `AE3dAttention`, `AE3dDilated`, `AE3dDilatedAttention`, `AE3dSeparableDilated`.

Protocol specifics:
- No new architecture implementation — just configure and run
- All trials commit (status = `REFERENCE`) — no FAILURE/revert
- Results go into the shared `trial_log.csv` with `trial_id`, `modification_description`, `delta_vs_champion` left blank
- Post-sweep: best avg_R2_mean model becomes the new champion in `EXPERIMENT.md`

### 7. Unexplored Directions list removed

Most directions from the list (strided conv, CBAM, GroupNorm, 1×1×1 compression, multi-scale) were already tried in experiment_architecture_1. Replaced with a **Strategy Reference** section pointing agents to either propose genuinely new ideas or revisit promising archive candidates (`archive/experiment_architecture_1/trial_log.csv`) under the new protocol.

### 8. Claude Code permissions configured

`/home/renku/work/.claude/settings.json` updated to pre-authorize all tool calls (fixing loop interruptions) while blocking dangerous operations:
- **Allow:** `Bash(*)`, `Edit(*)`, `Read(*)`, `Write(*)`
- **Deny:** `git push/pull/fetch/remote/reset`, `rm `, `rm -r*`

**Why this was needed:** the session settings file is at `/home/renku/work/.claude/settings.json` (session root), not at `SDSC-IRM/.claude/settings.json`. The SDSC-IRM settings were never loaded, causing every Edit/Bash call to prompt for authorization and break autonomous loop operation.
