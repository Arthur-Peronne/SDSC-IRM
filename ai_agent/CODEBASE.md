# CODEBASE.md

This document provides a technical overview of the `SDSC-IRM` codebase to assist in the iterative optimization of autoencoder (AE) models.

## 1. Available Architectures
Located in `src/models/ae_models.py`, the following models can be instantiated via `build_autoencoder`:

| Model Name | Type | Description |
| :--- | :--- | :--- |
| `AE3dCurrent` | Convolutional | Standard 3D AE with a flattened bottleneck. |
| `AE3dFCDeep` | Convolutional | Deeper architecture with progressive compression. |
| `AE3dConv` | Fully Conv | No linear layers; latent space is a small 3D tensor (requires `latent_dim` to be a multiple of 4). |
| `AE3dLinear` | Linear | Purely linear (no activations/convolutions); theoretically equivalent to PCA. |
| `AE3dFCDeep_VAE` | Variational | VAE version of `AE3dFCDeep` using the reparameterization trick. |

> Other architectures have been created by AI Agents during optimization trials. They are not listed here to avoid constant updates.

## 2. Optimization Levers
Parameters are managed via YAML configuration files (e.g., `configs/autoencoder.yaml`).

### Structural Levers
*   `model_name`: Selection of the architecture.
*   `latent_dimensions`: The size of the bottleneck/latent vector.

### Hyperparameter Levers
*   `lr`: Learning rate.
*   `weight_decay`: L2 regularization.
*   `dropout_rate`: Dropout probability.
*   `noise_std`: Denoising standard deviation.
*   `patience`: Early stopping patience.

### VAE-Specific Levers
*   `beta`: Weight of the KL-divergence term.
*   `beta_warmup_epochs`: Linear ramp-up period for $\beta$.

## 3. Operational Workflows

### Single Training & Evaluation
**Command:** `python scripts/run_autoencoder.py`
*   **CALC Mode:** Set `recalculate_ae: true` in the YAML.
*   **LOAD Mode:** Set `recalculate_ae: false` and provide a valid `load_run_id` (from MLflow).
*   **Tracking:** Results, metrics (MSE, KL, etc.), and models are logged to **MLflow**.

### Hyperparameter Optimization (Optuna)
**Command:** `python scripts/run_ae_optuna.py`
*   Uses `configs/ae_optuna.yaml` to define search ranges.
*   Results are stored in a SQLite database and logged to MLflow.
*   **Goal:** Find the best combination of hyperparameters for a specific architecture.

## 4. Key File Structure
*   `configs/`: YAML configuration files.
*   `scripts/`: Entry point scripts for training and optimization.
*   `src/`: Core logic (data loading, model definitions, training loops, tracking).
*   `mlruns/`: MLflow experiment tracking data.
*   `ai_agent/`: Agent-specific documentation and logs (residing in the root).
