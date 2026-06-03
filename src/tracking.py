# src/tracking.py
"""
MLflow utilities. All tracking interactions must go through this module.
Scripts must not import mlflow directly.

Naming convention
-----------------
experiment_name : pipeline type
    "autoencoder", "pca_spatial", "pca_temporal", "regression"

run_name : "{simulation_name}_{experiment_tag}"
    simulation_name identifies the data configuration:
        AE  → "AE3dFCDeep_240patients_split0_120dims"
        PCA → "PCA_240patients_split0_ED+ES"
    experiment_tag identifies the hyperparameter regime:
        AE  → "optuna", "baseline", ...
        PCA → "baseline"  (always — kept for symmetry with AE)

    Full run_name examples:
        "AE3dFCDeep_240patients_split0_120dims_optuna"
        "PCA_240patients_split0_ED+ES_baseline"
"""

import warnings
import mlflow
from contextlib import contextmanager

from src.config import MLRUNS_FOLDER


def _setup():
    warnings.filterwarnings("ignore", category=FutureWarning, module="mlflow")
    mlflow.set_tracking_uri(MLRUNS_FOLDER.as_uri())


@contextmanager
def start_run(experiment_name: str, run_name: str, tags: dict = None):
    """
    Context manager for an MLflow run.

    Usage
    -----
    with tracking.start_run("autoencoder", "AE3dFCDeep_240patients_split0_120dims_optuna"):
        tracking.log_params({"lr": 4.24e-5, "latent_dim": 120, ...})
        tracking.log_metric("train_loss", loss, step=epoch)
        tracking.log_model_state_dict(model)

    with tracking.start_run("pca_spatial", "PCA_240patients_split0_ED+ES_baseline"):
        tracking.log_params({"n_components": 240, "frame_tag": "ED+ES", ...})
        tracking.log_metrics({"r2_train": 0.95, "r2_test": 0.91})
    """
    _setup()
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name, tags=tags) as run:
        yield run

@contextmanager
def resume_run(run_id: str):
    """Resume an existing MLflow run to add new artifacts."""
    _setup()
    with mlflow.start_run(run_id=run_id) as run:
        yield run

def log_params(params: dict):
    """Log a dict of hyperparameters for the active run."""
    mlflow.log_params(params)


def log_metric(key: str, value: float, step: int = None):
    """Log a single metric, optionally at a given step (epoch)."""
    mlflow.log_metric(key, value, step=step)


def log_metrics(metrics: dict, step: int = None):
    """Log multiple metrics at once, optionally at a given step."""
    mlflow.log_metrics(metrics, step=step)


def log_model_state_dict(model, filename: str = "model.pth"):
    """Save a PyTorch state dict as an MLflow artifact."""
    import torch
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / filename
        torch.save(model.state_dict(), path)
        mlflow.log_artifact(str(path))


def log_artifact(path):
    """Log a file (plot, summary, etc.) as an MLflow artifact."""
    mlflow.log_artifact(str(path))


def log_sklearn_model(model, filename: str = "pca.joblib"):
    """Save a sklearn model as an MLflow artifact using joblib serialization."""
    import joblib
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / filename
        joblib.dump(model, path)
        mlflow.log_artifact(str(path))


def active_run_id() -> str:
    """Return the run_id of the currently active MLflow run."""
    _setup()
    run = mlflow.active_run()
    if run is None:
        raise RuntimeError("No active MLflow run")
    return run.info.run_id


def search_runs(experiment_name: str, filter_string: str = "", order_by: list = None):
    """
    Return a DataFrame of runs for the given experiment.

    Used by run_comparison_aepca.py to load results without parsing text files.

    Parameters
    ----------
    experiment_name : str
        e.g. "autoencoder" or "pca_spatial"
    filter_string : str
        MLflow filter syntax, e.g. "params.model_name = 'AE3dFCDeep'"
    order_by : list of str
        e.g. ["metrics.r2_test DESC"]
    """
    _setup()
    return mlflow.search_runs(
        experiment_names=[experiment_name],
        filter_string=filter_string,
        order_by=order_by or [],
    )
