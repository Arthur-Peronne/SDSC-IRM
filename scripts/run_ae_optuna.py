# scripts/run_ae_optuna.py
"""
Optuna hyperparameter optimisation for 3D autoencoders.

Reads configuration from configs/ae_optuna.yaml.
Results (plots, summary, SQLite DB) are tracked in MLflow under experiment "ae_optuna".

Usage:
    python scripts/run_ae_optuna.py

CALC mode: set plot_only: false — runs the Optuna study (resumes if DB already exists).
PLOT mode: set plot_only: true  — loads the existing study from SQLite and regenerates plots.

After finding the best hyperparameters, copy them into configs/autoencoder.yaml
and run run_autoencoder.py to train the final model.
"""

import yaml
import optuna
from pathlib import Path

from src.config import RESULTS_FOLDER
from src.data import loader
from src.training import ae_optuna as aeo
from src import tracking

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "ae_optuna.yaml"


def main():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    if cfg["n_val"] <= 0:
        raise ValueError("n_val must be > 0 for Optuna (early stopping requires a validation set)")

    frame_tag  = "ED+ES" if cfg["use_both_frames"] else cfg["frame_type"]
    n_train    = cfg["n_train"]
    n_val      = cfg["n_val"]
    study_name = cfg["study_name"]

    train_dataset, val_dataset, _, _, split_name = loader.load_tensor_datasets(
        source_folder="registered_frames",
        cache_folder="X_vectors",
        n_train=n_train,
        n_val=n_val,
        n_test=cfg["n_test"],
        special_split=cfg.get("special_split"),
        stratify_ongroup=cfg.get("stratify_ongroup", False),
        use_both_frames=cfg["use_both_frames"],
        frame_type=cfg["frame_type"],
        image_roi_only=cfg["image_roi_only"],
        mask=cfg["mask_ys"],
        binary_mask=cfg["mask_bin"],
        recalculate=False,
    )
    print(f"Data loaded | train: {len(train_dataset)} | val: {len(val_dataset)} | split: {split_name}")

    run_name = f"optuna_{study_name}_{split_name}"
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if cfg["plot_only"]:
        load_run_id = cfg.get("load_run_id")
        if not load_run_id:
            raise ValueError("plot_only: true requires load_run_id in the YAML")
        db_path = tracking.download_artifact(load_run_id)
        study = aeo.load_study(study_name, db_path)
        print(f"Study loaded from DB: {len(study.trials)} trials")

        with tracking.resume_run(load_run_id):
            tracking.log_artifact(CONFIG_PATH)
            _log_and_plot(study, cfg, n_train, n_val, split_name, frame_tag, study_name)
            
    else:
        mlflow_run, artifact_dir = tracking.start_run_and_get_id("ae_optuna", run_name)
        with mlflow_run:
            tracking.log_artifact(CONFIG_PATH)

            tracking.log_params({
                "model_name":        cfg["model_name"],
                "latent_dimensions": cfg["latent_dimensions"],
                "n_train":           n_train,
                "n_val":             n_val,
                "n_test":            cfg["n_test"],
                "split_name":        split_name,
                "frame_tag":         frame_tag,
                "stratify_ongroup":  cfg.get("stratify_ongroup", False),
                "image_roi_only":    cfg["image_roi_only"],
                "mask_ys":           cfg["mask_ys"],
                "mask_bin":          cfg["mask_bin"],
                "n_epochs":          cfg["n_epochs"],
                "batch_size":        cfg["batch_size"],
                "study_name":        study_name,
                "n_trials_target":   cfg["n_trials"],
            })
            # Hyperparameter fixed 
            fixed_hp_keys = {"lr", "weight_decay", "dropout_rate", "noise_std", "patience",
                            "beta", "beta_warmup_epochs"}
            fixed_params = {
                k: cfg[k] for k in fixed_hp_keys
                if k in cfg and k not in cfg.get("hp_ranges", {})
            }
            if fixed_params:
                tracking.log_params(fixed_params)

            db_path = artifact_dir / f"{study_name}.db"
            study = aeo.run_optuna(cfg, train_dataset, val_dataset, db_path)
            _log_and_plot(study, cfg, n_train, n_val, split_name, frame_tag, study_name)


def _log_and_plot(study, cfg, n_train, n_val, split_name, frame_tag, study_name):
    """Log metrics, plots and summary for a completed or loaded Optuna study."""
    n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    tracking.log_metric("best_val_loss",      study.best_value)
    tracking.log_metric("best_trial_number",  float(study.best_trial.number))
    tracking.log_metric("n_trials_completed", float(n_complete))
    for k, v in study.best_params.items():
        tracking.log_metric(f"best_{k}", float(v))

    plot_paths = aeo.plot_optuna_results(study, cfg, RESULTS_FOLDER)
    for p in plot_paths:
        tracking.log_artifact(p)

    summary_path = aeo.save_optuna_summary(study, RESULTS_FOLDER)
    tracking.log_artifact(summary_path)

    print(f"\nBest trial      : {study.best_trial.number}")
    print(f"Best val loss   : {study.best_value:.6f}")
    print("Best params :")
    for k, v in study.best_params.items():
        print(f"  {k:>20} = {v}")


if __name__ == "__main__":
    main()
