# scripts/run_autoencoder.py
"""
3D autoencoder training and evaluation on cardiac MRI data.

Reads configuration from configs/autoencoder.yaml.
Results (model + metrics + plots) are tracked in MLflow under experiment "autoencoder".

Usage:
    python scripts/run_autoencoder.py

LOAD mode: set recalculate_ae: false and load_run_id: <mlflow_run_id> in the YAML.
CALC mode: set recalculate_ae: true.
"""

import yaml
import mlflow
import torch
from pathlib import Path

from src.config import RESULTS_FOLDER
from src.data import loader
from src.models.ae_models import build_autoencoder
from src.training import ae_training as aet
from src.visualization import ae_plots as aep
from src import tracking

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "autoencoder.yaml"


def _run_one(cfg, model_name, latent_dimensions, split_name,
             n_train_images, n_val_images,
             train_dataset, val_dataset, test_dataset, X_maxnorm):
    """Train (or load) one AE model and log everything to a single MLflow run."""

    recalculate  = cfg["recalculate_ae"]
    load_run_id  = cfg.get("load_run_id") if not recalculate else None
    frame_tag    = "ED+ES" if cfg["use_both_frames"] else cfg["frame_type"]
    n_train      = cfg["n_train"]
    n_val        = cfg["n_val"]
    experiment_tag = cfg["experiment_tag"]

    if not recalculate and not load_run_id:
        raise ValueError("recalculate_ae: false requires load_run_id in YAML")

    run_name = (
        f"AE_{model_name}_{n_train_images}patients_{split_name}"
        f"_{latent_dimensions}dims_{experiment_tag}"
    )

    run_ctx = (
        tracking.start_run("autoencoder", run_name)
        if recalculate
        else tracking.resume_run(load_run_id)
    )

    with run_ctx:
        tracking.log_metric("latent_dim", latent_dimensions)

        # ── CALC mode ─────────────────────────────────────────────────────────
        if recalculate:
            tracking.log_artifact(CONFIG_PATH)
            tracking.log_params({
                "source_folder":    cfg["source_folder"],
                "model_name":       model_name,
                "latent_dimensions": latent_dimensions,
                "n_train":          n_train,
                "n_val":            n_val,
                "n_test":           cfg["n_test"],
                "split_name":       split_name,
                "stratify_ongroup": cfg.get("stratify_ongroup", False),
                "frame_tag":        frame_tag,
                "image_roi_only":   cfg["image_roi_only"],
                "mask_ys":          cfg["mask_ys"],
                "mask_bin":         cfg["mask_bin"],
                "experiment_tag":   experiment_tag,
                "n_epochs":         cfg["n_epochs"],
                "batch_size":       cfg["batch_size"],
                "lr":               cfg["lr"],
                "weight_decay":     cfg["weight_decay"],
                "dropout_rate":     cfg["dropout_rate"],
                "noise_std":        cfg["noise_std"],
                "patience":         cfg["patience"],
                "patience_scheduler": cfg["patience_scheduler"],
            })

            if n_val > 0:
                model, best_epoch, loss_history = aet.ae_training_early_stopping(
                    train_dataset=train_dataset,
                    validation_dataset=val_dataset,
                    model_name=model_name,
                    latent_dimensions=latent_dimensions,
                    n_epochs=cfg["n_epochs"],
                    batch_size=cfg["batch_size"],
                    lr=cfg["lr"],
                    patience=cfg["patience"],
                    patience_scheduler=cfg["patience_scheduler"],
                    weight_decay=cfg["weight_decay"],
                    dropout_rate=cfg["dropout_rate"],
                    noise_std=cfg["noise_std"],
                    beta=cfg["beta"],
                    beta_warmup_epochs=cfg["beta_warmup_epochs"],
                )
                tracking.log_params({"best_epoch": best_epoch})
            else:
                model, best_epoch, loss_history = aet.ae_training(
                    train_dataset=train_dataset,
                    model_name=model_name,
                    latent_dimensions=latent_dimensions,
                    n_epochs=cfg["n_epochs"],
                    batch_size=cfg["batch_size"],
                    lr=cfg["lr"],
                    weight_decay=cfg["weight_decay"],
                    dropout_rate=cfg["dropout_rate"],
                    noise_std=cfg["noise_std"],
                )
                best_epoch = cfg["n_epochs"]

            # ── Log loss history in MLflow ────────────────────────────────────────────
            for step, value in enumerate(loss_history.get("train", [])):
                tracking.log_metric("train_loss", value, step=step + 1)
            for step, value in enumerate(loss_history.get("validation", [])):
                tracking.log_metric("val_loss", value, step=step + 1)
            for step, value in enumerate(loss_history.get("train_mse", [])):
                tracking.log_metric("train_mse", value, step=step + 1)
            for step, value in enumerate(loss_history.get("train_kl", [])):
                tracking.log_metric("train_kl", value, step=step + 1)
            for step, value in enumerate(loss_history.get("val_mse", [])):
                tracking.log_metric("val_mse", value, step=step + 1)
            for step, value in enumerate(loss_history.get("val_kl", [])):
                tracking.log_metric("val_kl", value, step=step + 1)

            # Save model artifact in MLflow
            tracking.log_model_state_dict(model, filename=f"model_{best_epoch}epochs.pth")

            # Loss plot (always produced when training)
            loss_plot_path = RESULTS_FOLDER / f"{run_name}_train_val_loss.png"
            aep.plot_train_val_loss(
                loss_history=loss_history,
                best_epoch=best_epoch,
                simulation_name=run_name,
                save_path=loss_plot_path,
            )
            tracking.log_artifact(loss_plot_path)

            log_path = RESULTS_FOLDER / f"training_{experiment_tag}.log"
            if log_path.exists():
                tracking.log_artifact(log_path)

        # ── LOAD mode ─────────────────────────────────────────────────────────
        else:
            client = mlflow.MlflowClient()
            saved  = client.get_run(load_run_id).data.params
            expected = {
                "split_name":       split_name,
                "n_train":          str(n_train),
                "n_val":            str(n_val),
                "n_test":           str(cfg["n_test"]),
                "frame_tag":        frame_tag,
                "model_name":       model_name,
                "latent_dimensions": str(latent_dimensions),
                "source_folder":  cfg["source_folder"],
                "image_roi_only": str(cfg["image_roi_only"]),
                "mask_ys":        str(cfg["mask_ys"]),
                "mask_bin":       str(cfg["mask_bin"]),
            }
            mismatches = [
                f"  {k}: saved={saved.get(k)!r}, current={v!r}"
                for k, v in expected.items()
                if saved.get(k) != v
            ]
            if mismatches:
                raise ValueError(
                    f"Data/model mismatch with run {load_run_id}:\n" + "\n".join(mismatches)
                )

            best_epoch     = int(saved["best_epoch"])
            model_filename = f"model_{best_epoch}epochs.pth"
            local_path     = client.download_artifacts(load_run_id, model_filename)
            device         = aet.get_device()
            model          = build_autoencoder(
                model_name, latent_dimensions, dropout_rate=cfg["dropout_rate"]
            ).to(device)
            model.load_state_dict(torch.load(local_path, map_location=device))
            model.eval()
            loss_history = {"train": [], "validation": []}
            print(f"Model loaded from run {load_run_id} (best_epoch={best_epoch})")

        # ── Reconstruction metrics ─────────────────────────────────────────────
        if cfg["compute_metrics"]:
            datasets = [("train", train_dataset, 0)]
            if n_val > 0:
                datasets.append(("validation", val_dataset, n_train_images))
            datasets.append(("test", test_dataset, n_train_images + n_val_images))

            for metrics_dataset, dataset, offset in datasets:
                all_metrics = []
                for i, patient_tensor in enumerate(dataset):
                    x_true, x_pred = aet.ae_reconstructX(patient_tensor, X_maxnorm, model)
                    metrics = aet.reconstruction_metrics(
                        x_true=x_true,
                        x_pred=x_pred,
                        patient_number=offset + 1 + i,
                    )
                    all_metrics.append(metrics)

                summary = aet.ae_aggregate_metrics(all_metrics)
                for metric, stats in summary.items():
                    for stat, value in stats.items():
                        tracking.log_metric(f"{metrics_dataset}_{metric}_{stat}", value)

                print(
                    f"[{metrics_dataset}] R2 mean={summary['R2']['mean']:.4f} "
                    f"std={summary['R2']['std']:.4f} | "
                    f"MSE mean={summary['MSE']['mean']:.6f}"
                )

        # ── Reconstruction plots ───────────────────────────────────────────────
        if cfg["plot_reconstruction"]:
            n_development = n_train + n_val

            if cfg["recons_auto"]:
                # Compute metrics on test set at this latent_dim for patient selection
                test_metrics = []
                for i, patient_tensor in enumerate(test_dataset):
                    x_true, x_pred = aet.ae_reconstructX(patient_tensor, X_maxnorm, model)
                    test_metrics.append(aet.reconstruction_metrics(x_true, x_pred, n_train_images + n_val_images + 1 + i))
                selected = aep.ae_select_representative_patients(
                    test_metrics,
                    use_both_frames=cfg["use_both_frames"],
                    n_train_images=n_train_images,
                    n_val_images=n_val_images,
                    n_development=n_development,
                )
                patients_torecons = [(v["real_patient"], v["frame_type"]) for v in selected.values()]
            else:
                patients_torecons = [tuple(p) for p in cfg["patients_torecons_manual"]]

            aep.ae_plotcompare_selected(
                patients_torecons=patients_torecons,
                use_both_frames=cfg["use_both_frames"],
                n_development=n_development,
                n_train_images=n_train_images,
                n_val_images=n_val_images,
                train_dataset=train_dataset,
                validation_dataset=val_dataset,
                test_dataset=test_dataset,
                X_maxnorm=X_maxnorm,
                model=model,
                model_name=model_name,
                split_name=split_name,
                latent_dimensions=latent_dimensions,
                n_epochs=best_epoch,
            )


def main():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    # ── Float fix ────────────────────────────────────────────────────
    for key in ("lr", "weight_decay", "dropout_rate", "noise_std", "beta"):
        cfg[key] = float(cfg[key])
    
    # ── Derived parameters ────────────────────────────────────────────────────
    n_train         = cfg["n_train"]
    n_val           = cfg["n_val"]
    use_both_frames = cfg["use_both_frames"]
    n_train_images  = n_train * 2 if use_both_frames else n_train
    n_val_images    = n_val   * 2 if use_both_frames else n_val

    # ── Load data (before MLflow — split_name needed for run_name) ────────────
    train_dataset, val_dataset, test_dataset, X_maxnorm, split_name = loader.load_tensor_datasets(
        source_folder=cfg["source_folder"],
        cache_folder=cfg["cache_folder"],
        n_train=n_train,
        n_val=n_val,
        n_test=cfg["n_test"],
        special_split=cfg.get("special_split"),
        stratify_ongroup=cfg.get("stratify_ongroup", False),
        use_both_frames=use_both_frames,
        frame_type=cfg["frame_type"],
        image_roi_only=cfg["image_roi_only"],
        mask=cfg["mask_ys"],         
        binary_mask=cfg["mask_bin"], 
        recalculate=cfg["recalculate_x"],
    )
    print(
        f"Data loaded | train: {len(train_dataset)} | "
        f"val: {len(val_dataset) if val_dataset else 0} | "
        f"test: {len(test_dataset)} | split: {split_name}"
    )

    # ── Run ───────────────────────────────────────────────────────────────────
    if cfg["multiple_models_and_dims"]:
        if not cfg["recalculate_ae"]:
            raise ValueError("multiple_models_and_dims: true requires recalculate_ae: true")
        for model_name in cfg["models_list"]:
            for latent_dimensions in cfg["latdim_list"]:
                print(f"\n{'='*60}\nModel: {model_name} | latent_dim: {latent_dimensions}\n{'='*60}")
                _run_one(
                    cfg, model_name, latent_dimensions, split_name,
                    n_train_images, n_val_images,
                    train_dataset, val_dataset, test_dataset, X_maxnorm,
                )
    else:
        _run_one(
            cfg, cfg["model_name"], cfg["latent_dimensions"], split_name,
            n_train_images, n_val_images,
            train_dataset, val_dataset, test_dataset, X_maxnorm,
        )

if __name__ == "__main__":
    main()
