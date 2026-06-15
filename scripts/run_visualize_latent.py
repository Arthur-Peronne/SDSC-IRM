# scripts/run_visualize_latent.py
"""
Visualize the latent space of a trained AE or VAE model.

Loads a model from an MLflow run, encodes all patients,
projects latent vectors to 2D (PCA), colored by ACDC group.
The plot is logged as artifact in the original MLflow run.

Usage:
    python scripts/run_visualize_latent.py
"""

import yaml
import torch
import mlflow
from pathlib import Path

from src.config import RESULTS_FOLDER
from src.data import loader
from src.data.importdata import load_acdc_groups
from src.models.ae_models import build_autoencoder
from src.training import ae_training as aet
from src.visualization import ae_plots as aep
from src import tracking

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "visualize_latent.yaml"


def main():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    load_run_id = cfg["load_run_id"]
    if not load_run_id:
        raise ValueError("load_run_id must be set in visualize_latent.yaml")

    # ── Load run params from MLflow ───────────────────────────────────────────
    tracking._setup()
    client = mlflow.MlflowClient()
    run       = client.get_run(load_run_id)
    params    = run.data.params
    run_name  = run.info.run_name

    model_name        = params["model_name"]
    latent_dimensions = int(params["latent_dimensions"])
    dropout_rate      = float(params.get("dropout_rate", 0.0))
    best_epoch        = int(params["best_epoch"])
    print(f"Run: {run_name} | {model_name} | latent_dim={latent_dimensions} | epoch={best_epoch}")

    # ── Load model ────────────────────────────────────────────────────────────
    device     = aet.get_device()
    local_path = client.download_artifacts(load_run_id, f"model_{best_epoch}epochs.pth")
    model      = build_autoencoder(model_name, latent_dimensions,
                                   dropout_rate=dropout_rate).to(device)
    model.load_state_dict(torch.load(local_path, map_location=device))
    model.eval()

    # ── Load data ─────────────────────────────────────────────────────────────
    n_patients      = cfg["n_patients"]
    use_both_frames = cfg["use_both_frames"]

    all_dataset, _, _, _, _ = loader.load_tensor_datasets(
        source_folder="registered_frames",
        cache_folder="X_vectors",
        n_train=n_patients,
        n_val=0,
        n_test=0,
        special_split=None,
        use_both_frames=use_both_frames,
        frame_type="ED",
        image_roi_only=True,
        mask=False,
        binary_mask=False,
        recalculate=False,
    )

    # ── Labels ────────────────────────────────────────────────────────────────
    groups = load_acdc_groups(n_patients)
    labels = groups + groups if use_both_frames else groups

    # ── Encode ────────────────────────────────────────────────────────────────
    print("Encoding all patients...")
    mus = aep.collect_latent_vectors(model, all_dataset, device)
    print(f"Latent vectors: {mus.shape}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    projection = cfg["projection"]

    if projection == "pca":
        save_path = RESULTS_FOLDER / f"{run_name}_latent_pca.png"
        aep.plot_latent_pca(mus, labels, run_name, use_both_frames, save_path)
    elif projection == "tsne":
        perplexity = cfg["perplexity"]
        save_path = RESULTS_FOLDER / f"{run_name}_latent_tsne_p{perplexity}.png"
        aep.plot_latent_tsne(mus, labels, run_name, use_both_frames, perplexity, save_path)
    else:
        raise ValueError(f"Unknown projection: {projection!r} (expected 'pca' or 'tsne')")

    # ── Log dans MLflow ───────────────────────────────────────────────────────
    with tracking.resume_run(load_run_id):
        tracking.log_artifact(save_path)
    print(f"Logged to MLflow run {load_run_id}")


if __name__ == "__main__":
    main()