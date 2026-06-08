# src/training/ae_training.py
"""
Training and evaluation utilities for the 3D autoencoder.

Design principle: these functions contain only training/evaluation logic.
All file I/O (model saving, metric logging) is handled by run_autoencoder.py via MLflow,
following the same pattern as run_pca_spatial.py.
"""

import tempfile
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path

from src.config import RESULTS_FOLDER
from src.models.ae_models import build_autoencoder


def get_device(verbose=False):
    """Return the best available torch device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        if verbose:
            print("Using CPU")
    return device


def ae_training(
    train_dataset,
    model_name,
    latent_dimensions,
    n_epochs=75,
    batch_size=1,
    lr=1e-5,
    weight_decay=0.0,
    dropout_rate=0.0,
    noise_std=0.0,
):
    """
    Train the 3D autoencoder for a fixed number of epochs (no early stopping).
    Used when no validation set is available.

    Returns
    -------
    model : nn.Module  (eval mode)
    n_epochs : int
    loss_history : dict  {"train": [float, ...]}
    """
    device = get_device()
    model = build_autoencoder(model_name, latent_dimensions, dropout_rate=dropout_rate).to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_history = {"train": []}

    print(
        f"Training {model_name} | device={device} | "
        f"batch_size={batch_size} | n_epochs={n_epochs} | lr={lr:.2e} | "
        f"weight_decay={weight_decay:.2e} | dropout={dropout_rate:.2f} | "
        f"noise_std={noise_std:.4f}"
    )

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0

        for (x_batch,) in train_loader:
            x_batch = x_batch.to(device, non_blocking=(device.type == "cuda"))

            if noise_std > 0.0:
                x_noisy = torch.clamp(x_batch + torch.randn_like(x_batch) * noise_std, 0.0, 1.0)
            else:
                x_noisy = x_batch

            optimizer.zero_grad()
            output = model(x_noisy)
            x_recon = output[0]
            loss = criterion(x_recon, x_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        loss_history["train"].append(avg_loss)
        print(f"Epoch {epoch + 1}/{n_epochs} | train: {avg_loss:.6f}")

    model.eval()
    return model, n_epochs, loss_history


def ae_reconstructX(patient_tensor, X_maxnorm, model):
    """
    Reconstruct one patient volume with the trained model.

    Returns x_true and x_pred, both denormalized, shape (128, 128, 32).
    """
    device = next(model.parameters()).device
    x_in = patient_tensor[0].unsqueeze(0).to(device)
    model.eval()

    with torch.no_grad():
        output = model(x_in)
        x_recon = output[0]

    x_true_np = x_in.squeeze(0).squeeze(0).detach().cpu().numpy()      # (32, 128, 128)
    x_true_np = np.transpose(x_true_np, (1, 2, 0)) * X_maxnorm         # (128, 128, 32)

    x_recon_np = x_recon.squeeze(0).squeeze(0).detach().cpu().numpy()   # (32, 128, 128)
    x_recon_np = np.transpose(x_recon_np, (1, 2, 0)) * X_maxnorm        # (128, 128, 32)

    return x_true_np, x_recon_np


def reconstruction_metrics(x_true, x_pred, patient_number):
    """
    Compute reconstruction metrics between two 3D arrays of identical shape.

    Returns
    -------
    dict : {"MSE", "RMSE", "MAE", "R2", "patient_number"}
    """
    if x_true.shape != x_pred.shape:
        raise ValueError(f"Shape mismatch: {x_true.shape} vs {x_pred.shape}")

    diff   = x_true - x_pred
    mse    = float(np.mean(diff ** 2))
    rmse   = float(np.sqrt(mse))
    mae    = float(np.mean(np.abs(diff)))
    ss_res = np.sum(diff ** 2)
    ss_tot = np.sum((x_true - np.mean(x_true)) ** 2)
    r2     = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2, "patient_number": int(patient_number)}


def ae_aggregate_metrics(all_metrics):
    """
    Aggregate per-patient reconstruction metrics.

    Returns
    -------
    dict : {metric_name: {"mean", "std", "min", "max", "median"}}
    """
    summary = {}
    for metric_name in ("MSE", "RMSE", "MAE", "R2"):
        values = np.array([m[metric_name] for m in all_metrics], dtype=np.float32)
        summary[metric_name] = {
            "mean":   float(np.mean(values)),
            "std":    float(np.std(values)),
            "min":    float(np.min(values)),
            "max":    float(np.max(values)),
            "median": float(np.median(values)),
        }
    return summary


def _compute_validation_loss(model, validation_dataset, batch_size, device, criterion, beta=1.0):
    """Compute mean reconstruction loss on the validation set (no gradients)."""
    val_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )

    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for (x_batch,) in val_loader:
            x_batch = x_batch.to(device, non_blocking=(device.type == "cuda"))
            output = model(x_batch)
            if len(output) == 4:
                x_recon, _, mu, logvar = output
                loss, _, _ = _vae_loss(x_recon, x_batch, mu, logvar, beta)
            else:
                x_recon, _ = output
                loss = criterion(x_recon, x_batch)
            total_loss += loss.item()

    return total_loss / len(val_loader)


def _vae_loss(x_recon, x_target, mu, logvar, beta):
    """VAE loss = MSE reconstruction + beta * KL divergence."""
    mse = nn.functional.mse_loss(x_recon, x_target)
    kl  = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return mse + beta * kl, mse, kl


def ae_training_early_stopping(
    train_dataset,
    validation_dataset,
    model_name,
    latent_dimensions,
    n_epochs=300,
    batch_size=1,
    lr=1e-3,
    patience=40,
    patience_scheduler=None,
    weight_decay=0.0,
    dropout_rate=0.0,
    noise_std=0.0,
    beta=1.0,
    beta_warmup_epochs=20,
):
    """
    Train the 3D autoencoder with early stopping based on validation loss.

    The best model is saved to a temporary file during training and loaded
    at the end. Permanent storage (MLflow artifact) is the caller's responsibility.

    Returns
    -------
    model : nn.Module  (best model, eval mode)
    best_epoch : int
    loss_history : dict  {"train": [...], "validation": [...]}
    """
    if patience_scheduler is None:
        patience_scheduler = patience // 5

    device = get_device()
    model = build_autoencoder(model_name, latent_dimensions, dropout_rate=dropout_rate).to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda"),
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=patience_scheduler,
    )

    loss_history = {"train": [], "validation": [], "train_mse": [], "train_kl": []}
    best_val_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0

    # Temporary file to store the best model state during training
    tmp = tempfile.NamedTemporaryFile(suffix=".pth", delete=False)
    temp_best_path = Path(tmp.name)
    tmp.close()

    print(
        f"Training {model_name} | device={device} | "
        f"batch_size={batch_size} | patience={patience} | max_epochs={n_epochs}"
    )

    for epoch in range(n_epochs):

        beta_current = (
            min(beta, beta * (epoch + 1) / beta_warmup_epochs)
            if beta_warmup_epochs > 0 else beta
        )

        # ── Training pass ─────────────────────────────────────────────────────
        model.train()
        epoch_train_loss = 0.0             
        epoch_train_mse = 0.0
        epoch_train_kl  = 0.0

        for (x_batch,) in train_loader:
            x_batch = x_batch.to(device, non_blocking=(device.type == "cuda"))

            if noise_std > 0.0:
                x_noisy = torch.clamp(x_batch + torch.randn_like(x_batch) * noise_std, 0.0, 1.0)
            else:
                x_noisy = x_batch

            optimizer.zero_grad()
            output = model(x_noisy)
            if len(output) == 4:
                x_recon, _, mu, logvar = output
                loss, mse, kl = _vae_loss(x_recon, x_batch, mu, logvar, beta_current)
                epoch_train_mse += mse.item()
                epoch_train_kl  += kl.item()
            else:
                x_recon, _ = output
                loss = criterion(x_recon, x_batch)

            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_train_mse = epoch_train_mse / len(train_loader)
        avg_train_kl  = epoch_train_kl  / len(train_loader)
        loss_history["train"].append(avg_train_loss)
        loss_history["train_mse"].append(avg_train_mse)
        loss_history["train_kl"].append(avg_train_kl)

        # ── Validation pass ───────────────────────────────────────────────────
        avg_val_loss = _compute_validation_loss(
            model, validation_dataset, batch_size, device, criterion, beta=beta_current,
        )
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        loss_history["validation"].append(avg_val_loss)

        current_epoch = epoch + 1
        is_best = avg_val_loss < best_val_loss
        print(
            f"Epoch {current_epoch}/{n_epochs} "
            f"| train: {avg_train_loss:.6f} | val: {avg_val_loss:.6f} | lr: {current_lr:.2e}"
            + (" ✓ best" if is_best else f" (no improvement for {epochs_without_improvement + 1} epochs)")
        )

        if is_best:
            best_val_loss = avg_val_loss
            best_epoch = current_epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), temp_best_path)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(
                f"\nEarly stopping at epoch {current_epoch}. "
                f"Best epoch: {best_epoch} (val loss: {best_val_loss:.6f})."
            )
            break

    # Load best weights, clean up temp file
    model.load_state_dict(torch.load(temp_best_path, map_location=device))
    temp_best_path.unlink(missing_ok=True)
    model.eval()

    return model, best_epoch, loss_history


def get_best_epochs_stats_from_mlflow(model_name, split_name, experiment_tag, latdim_list):
    """
    Retrieve best_epoch values from MLflow across latent dims for a given
    model / split / experiment combination.

    Requires that each run logged `best_epoch` as an MLflow param
    (done by run_autoencoder.py after training with early stopping).

    Returns
    -------
    dict : {latent_dim: {"epochs": [...], "mean", "median", "std", "min", "max"}}
    """
    from src import tracking

    df = tracking.search_runs(
        experiment_name="autoencoder",
        filter_string=(
            f"params.model_name = '{model_name}' and "
            f"params.split_name = '{split_name}' and "
            f"params.experiment_tag = '{experiment_tag}'"
        ),
    )

    if df.empty:
        return {}

    stats = {}
    for latent_dim in latdim_list:
        subset = df[df["params.latent_dimensions"] == str(latent_dim)]
        if subset.empty:
            continue
        epochs = [int(e) for e in subset["params.best_epoch"].dropna()]
        if not epochs:
            continue
        stats[latent_dim] = {
            "epochs": epochs,
            "mean":   float(np.mean(epochs)),
            "median": float(np.median(epochs)),
            "std":    float(np.std(epochs)),
            "min":    int(np.min(epochs)),
            "max":    int(np.max(epochs)),
        }
    return stats


def print_and_save_best_epochs_stats(stats, model_name, experiment_tag, results_folder_save=None):
    """
    Print and save best_epoch stats from get_best_epochs_stats_from_mlflow.
    One row per latent_dim, aggregated stats at the bottom.

    Parameters
    ----------
    stats : dict
        Output of get_best_epochs_stats_from_mlflow.
        Keys are latent_dim (int).
    """
    results_folder_save = Path(results_folder_save) if results_folder_save else RESULTS_FOLDER
    all_epochs = [e for s in stats.values() for e in s["epochs"]]

    lines = []
    lines.append(f"Best epoch stats — {model_name} | {experiment_tag}")
    lines.append("=" * 60)
    lines.append(f"{'latent_dim':>12} {'epochs':>30}")
    lines.append("-" * 44)

    for latent_dim, s in sorted(stats.items()):
        lines.append(f"{latent_dim:>12} {str(s['epochs']):>30}")

    lines.append("")
    lines.append("Aggregated stats across all latent dims")
    lines.append("-" * 60)
    if all_epochs:
        lines.append(
            f"mean={np.mean(all_epochs):.1f}  median={np.median(all_epochs):.1f}  "
            f"std={np.std(all_epochs):.1f}  min={min(all_epochs)}  max={max(all_epochs)}"
        )
    lines.append("=" * 60)

    for line in lines:
        print(line)

    save_path = results_folder_save / f"best_epochs_stats_{model_name}_{experiment_tag}.txt"
    with open(save_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nSaved: {save_path}")
