# src/training/ae_optuna.py
"""
Optuna hyperparameter optimisation for 3D autoencoders.

Each trial trains one AE with early stopping and returns the best validation loss.
Results are persisted in a SQLite DB so the study can be interrupted and resumed.
All I/O (MLflow logging, artifact saving) is handled by run_ae_optuna.py.
"""

import functools
from pathlib import Path

import optuna
import optuna.trial as optuna_trial
import matplotlib.pyplot as plt

from src.training import ae_training as aet


def objective(
    trial: optuna.Trial,
    config: dict,
    train_dataset,
    val_dataset,
) -> float:
    """Sample one set of hyperparams and return the best validation loss."""
    hp = {}
    for name, bounds in config["hp_ranges"].items():
        # Skip VAE-specific params if model is not VAE
        if name in ("beta", "beta_warmup_epochs") and config["model_name"] != "AE3dFCDeep_VAE":
            continue
        if bounds["type"] == "int":
            hp[name] = trial.suggest_int(name, int(bounds["low"]), int(bounds["high"]))
        else:
            hp[name] = trial.suggest_float(
                name,
                float(bounds["low"]),
                float(bounds["high"]),
                log=bounds.get("log", False),
            )

    try:
        _, best_epoch, loss_history = aet.ae_training_early_stopping(
            train_dataset=train_dataset,
            validation_dataset=val_dataset,
            model_name=config["model_name"],
            latent_dimensions=config["latent_dimensions"],
            n_epochs=config["n_epochs"],
            batch_size=config["batch_size"],
            lr=hp.get("lr", 1e-5),
            patience=hp.get("patience", 40),
            patience_scheduler=None,
            weight_decay=hp.get("weight_decay", 0.0),
            dropout_rate=hp.get("dropout_rate", 0.0),
            noise_std=hp.get("noise_std", 0.0),
            beta=hp.get("beta", 1.0),
            beta_warmup_epochs=hp.get("beta_warmup_epochs", 20),
        )
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        raise optuna.exceptions.TrialPruned()

    best_val_mse = min(loss_history["val_mse"])
    trial.set_user_attr("best_epoch", best_epoch)

    hp_str = " | ".join(
        f"{k}={v:.2e}" if isinstance(v, float) else f"{k}={v}"
        for k, v in hp.items()
    )
    print(f"Trial {trial.number:>3} | val_mse_loss={best_val_mse:.6f} | epoch={best_epoch:>4} | {hp_str}")
    return best_val_mse


def load_study(study_name: str, db_path: Path) -> optuna.Study:
    return optuna.load_study(
        study_name=study_name,
        storage=f"sqlite:///{db_path}",
    )


def run_optuna(
    config: dict,
    train_dataset,
    val_dataset,
    db_path: Path,
) -> optuna.Study:
    db_path.parent.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        direction="minimize",
        study_name=config["study_name"],
        storage=f"sqlite:///{db_path}",
        load_if_exists=True,
    )

    if len(study.trials) == 0 and config.get("hp_initial"):
        study.enqueue_trial(config["hp_initial"])

    n_done = len([t for t in study.trials if t.state == optuna_trial.TrialState.COMPLETE])
    n_remaining = max(0, config["n_trials"] - n_done)
    print(f"Trials complétés : {n_done} / {config['n_trials']} — restants : {n_remaining}")

    if n_remaining > 0:
        obj = functools.partial(
            objective,
            config=config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
        )
        study.optimize(obj, n_trials=n_remaining, show_progress_bar=True)

    return study


_COLORS = [
    "#534AB7", "#1D9E75", "#D85A30", "#BA7517",
    "#378ADD", "#E24B4A", "#8B4513", "#9467bd",
]


def plot_optuna_results(study: optuna.Study, config: dict, save_dir: Path) -> list[Path]:
    """
    Generate and save two plots: optimization history and hyperparameter evolution.
    Returns the list of saved file paths.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    completed.sort(key=lambda t: t.number)

    if not completed:
        print("No completed trials — skipping plots.")
        return []

    trial_nums = [t.number for t in completed]
    val_losses = [t.value for t in completed]
    best_trial = trial_nums[val_losses.index(min(val_losses))]
    study_name = study.study_name
    sorted_losses = sorted(val_losses, reverse=True)
    ylim_top = sorted_losses[1] * 1.02 if len(sorted_losses) > 1 else max(val_losses) * 1.05
    saved_paths = []

    # ── Plot 1: Optimization history ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(trial_nums, val_losses, color="#B4B2A9", linewidth=1,
            marker="o", markersize=3, label="Val loss")
    ax.axvline(x=best_trial, color="#E24B4A", linewidth=1.2,
               linestyle="--", label=f"Best trial ({best_trial})")
    ax.scatter([best_trial], [min(val_losses)], color="#E24B4A", s=60, zorder=5)
    ax.set_xlabel("Trial number")
    ax.set_ylabel("Val MSE (reconstruction loss)")
    ax.set_title(f"Optuna — optimization history\n{study_name}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=min(val_losses) * 0.95, top=ylim_top)
    fig.tight_layout()
    p1 = save_dir / f"optuna_{study_name}_history.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print(f"Saved: {p1}")
    saved_paths.append(p1)

    # ── Plot 2: Hyperparameter evolution ───────────────────────────────────────
    hp_names = list(config["hp_ranges"].keys())
    n_hp = len(hp_names)
    fig, axes = plt.subplots(n_hp, 1, figsize=(10, 3 * n_hp), sharex=True)
    if n_hp == 1:
        axes = [axes]

    for ax, hp_name, color in zip(axes, hp_names, _COLORS):
        ax_vl = ax.twinx()
        ax_vl.plot(trial_nums, val_losses, color="#B4B2A9",
                   linewidth=0.8, linestyle="--", alpha=0.5)
        ax_vl.set_ylim(bottom=min(val_losses) * 0.95, top=ylim_top)
        ax_vl.set_ylabel("val loss", fontsize=8, color="#B4B2A9")
        ax_vl.tick_params(axis="y", labelcolor="#B4B2A9", labelsize=7)

        values = [t.params.get(hp_name) for t in completed]
        ax.plot(trial_nums, values, color=color, linewidth=1.5, marker="o", markersize=3)
        ax.axvline(x=best_trial, color="#E24B4A", linewidth=1.2, linestyle="--")
        ax.set_ylabel(hp_name, fontsize=9)
        ax.grid(True, alpha=0.3)

        if config["hp_ranges"][hp_name].get("log", False):
            ax.set_yscale("log")

    axes[-1].set_xlabel("Trial number")
    fig.suptitle(f"Optuna — hyperparameter evolution\n{study_name}", fontsize=11)
    fig.tight_layout()
    p2 = save_dir / f"optuna_{study_name}_hyperparams.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    print(f"Saved: {p2}")
    saved_paths.append(p2)

# ── Plot 3: Val loss vs hyperparameter value ────────────────────────────────
    fig, axes = plt.subplots(n_hp, 1, figsize=(10, 3 * n_hp), sharex=False)
    if n_hp == 1:
        axes = [axes]

    best_val_loss = min(val_losses)

    for ax, hp_name, color in zip(axes, hp_names, _COLORS):
        values = [t.params.get(hp_name) for t in completed]
        is_best = [v == best_val_loss for v in val_losses]

        ax.scatter(
            [v for v, b in zip(values, is_best) if not b],
            [l for l, b in zip(val_losses, is_best) if not b],
            color=color, alpha=0.7, s=30, zorder=3,
        )
        ax.scatter(
            [v for v, b in zip(values, is_best) if b],
            [l for l, b in zip(val_losses, is_best) if b],
            color="#E24B4A", s=80, zorder=5, label="Best trial",
        )
        ax.axhline(y=best_val_loss, color="#E24B4A", linewidth=1.2,
                   linestyle="--", alpha=0.7)
        ax.set_xlabel(hp_name, fontsize=9)
        ax.set_ylabel("val MSE", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        if config["hp_ranges"][hp_name].get("log", False):
            ax.set_xscale("log")

    fig.suptitle(f"Optuna — val MSE vs hyperparameter value\n{study_name}", fontsize=11)
    fig.tight_layout()
    p3 = save_dir / f"optuna_{study_name}_hp_vs_loss.png"
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    print(f"Saved: {p3}")
    saved_paths.append(p3)
    
    return saved_paths


def save_optuna_summary(study: optuna.Study, save_dir: Path) -> Path:
    """Save a human-readable text summary of the study. Returns the file path."""
    save_dir.mkdir(parents=True, exist_ok=True)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    hp_names = list(completed[0].params.keys()) if completed else []

    lines = []
    lines.append("=" * 60)
    lines.append(f"Optuna Study : {study.study_name}")
    lines.append(f"Trials completed : {len(completed)}")
    lines.append(f"Best val MSE    : {study.best_value:.6f}")
    lines.append("")
    lines.append("Best hyperparameters :")
    for k, v in study.best_params.items():
        lines.append(f"  {k:>20} = {v}")
    lines.append("")

    try:
        importance = optuna.importance.get_param_importances(study)
        lines.append("Hyperparameter importance :")
        for k, v in importance.items():
            bar = "█" * int(v * 30)
            lines.append(f"  {k:>20} : {v:.3f}  {bar}")
        lines.append("")
    except Exception:
        lines.append("(importance unavailable — too few trials)")
        lines.append("")

    header = f"{'Trial':>6} {'Val loss':>12} {'Epoch':>6} " + " ".join(f"{h:>12}" for h in hp_names)
    lines.append(header)
    lines.append("-" * len(header))
    for t in sorted(completed, key=lambda t: t.value or float("inf")):
        row = (
            f"{t.number:>6} {t.value:>12.6f} "
            f"{t.user_attrs.get('best_epoch', '?'):>6} "
            + " ".join(
                f"{t.params.get(h, 0):>12.2e}" if isinstance(t.params.get(h), float)
                else f"{t.params.get(h, 0):>12}"
                for h in hp_names
            )
        )
        lines.append(row)
    lines.append("=" * 60)

    out_path = save_dir / f"optuna_{study.study_name}_summary.txt"
    out_path.write_text("\n".join(lines))
    print(f"Summary saved: {out_path}")
    return out_path
