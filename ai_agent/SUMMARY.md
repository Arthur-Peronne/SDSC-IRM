# 📜 PROJECT OPERATIONAL SUMMARY: SDSC-IRM AE OPTIMIZATION

> **This file is structured chronologically.** It documents the decisions, reasoning, and outcomes of each conversation session. Read from top to bottom to understand the project's evolution. For the current state of rules and constraints, see `AGENT.md`.

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

## 📅 SESSION LOG — 2026-06-19 (Continued)

### 8. Protocol Refinement & Optimization Strategy
- **Hyperparameter Policy:** Decided that for the upcoming 20-trial architecture optimization loop, all hyperparameters (learning rate, weight decay, etc.) will be **FROZEN**. This ensures that improvements in `val_mse` are directly attributable to architectural changes, maintaining scientific rigor and maximizing iteration speed.
- **Hypothesis-Driven Experimentation:** Refined `EXPERIMENT.md` to transition from "idea-based" to "hypothesis-driven" research. The agent will now explicitly formulate a hypothesis (What, Why, How) and design the implementation based on whether it is performing **Exploration** (new families) or **Exploitation** (refining the current champion).
- **Simulated Success for Validation:** Established a preliminary test protocol (`EXPERIMENT_existingarchitectures.md`) to validate the agent's ability to iterate through all existing models in `ae_models.py` using a "simulated success" commit strategy to preserve all results.
- **Workflow Improvements:** Updated the commit/revert protocol to use `git checkout HEAD -- <file>` for more robust configuration management.
