# 📜 PROJECT OPERATIONAL SUMMARY: SDSC-IRM AE OPTIMIZATION

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
