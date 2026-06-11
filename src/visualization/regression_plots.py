# src/visualization/regression_plots.py
"""
Plotting functions for logistic and linear regression results.
All functions take in-memory result dicts — no file parsing.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.config import RESULTS_FOLDER


def plot_logistic_metrics(results: list[dict], y_name: str, run_label: str,
                          binary: bool = False) -> Path:
    """
    Plot classification metrics vs number of latent dimensions.

    Parameters
    ----------
    results : list of dict
        Each dict from eval_logistic_binary or eval_logistic_multiclass,
        sorted by n_dims.
    y_name : str
        Target variable name (used in title).
    run_label : str
        Short label for the run (used in filename and title).
    binary : bool
        True → binary metrics (AUC, precision, recall).
        False → multiclass metrics (accuracy, precision_macro, recall_macro).

    Returns
    -------
    Path to saved figure.
    """
    n_dims = [r["n_dims"] for r in results]
    accuracy = [r["accuracy"] for r in results]
    ev = [r["explained_variance"] for r in results]

    if binary:
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(8, 10), sharex=True)

        ax1.plot(n_dims, [r["roc_auc"] for r in results], marker="o", label="ROC AUC")
        ax1.plot(n_dims, accuracy, marker="o", label="Accuracy")
        ax1.plot(n_dims, ev, linestyle="--", color="gray", alpha=0.35,
                 label="Cum. explained variance")
        ax1.set_ylabel("Score")
        ax1.set_title(f"{run_label} — {y_name} (binary) — Overall")
        ax1.set_ylim(0, 1.05)
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2.plot(n_dims, [r["recall"] for r in results], marker="o", label="Recall")
        ax2.plot(n_dims, [r["precision"] for r in results], marker="o", label="Precision")
        ax2.plot(n_dims, ev, linestyle="--", color="gray", alpha=0.35,
                 label="Cum. explained variance")
        ax2.set_xlabel("Latent dimensions")
        ax2.set_ylabel("Score")
        ax2.set_title(f"{run_label} — {y_name} (binary) — Positive class")
        ax2.set_ylim(0, 1.05)
        ax2.grid(True, alpha=0.3)
        ax2.legend()

    else:
        classes = results[0]["classes"]
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(9, 14), sharex=True)

        ax1.plot(n_dims, accuracy, marker="o", label="Accuracy")
        ax1.plot(n_dims, ev, linestyle="--", color="gray", alpha=0.35,
                 label="Cum. explained variance")
        ax1.set_ylabel("Score")
        ax1.set_title(f"{run_label} — {y_name} — Accuracy")
        ax1.set_ylim(0, 1.05)
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        for cls in classes:
            vals = [r["recall_per_class"].get(cls, 0.0) for r in results]
            ax2.plot(n_dims, vals, marker="o", label=f"Recall {cls}")
        ax2.plot(n_dims, ev, linestyle="--", color="gray", alpha=0.35,
                 label="Cum. explained variance")
        ax2.set_ylabel("Score")
        ax2.set_title(f"{run_label} — {y_name} — Recall per class")
        ax2.set_ylim(0, 1.05)
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        for cls in classes:
            vals = [r["precision_per_class"].get(cls, 0.0) for r in results]
            ax3.plot(n_dims, vals, marker="o", label=f"Precision {cls}")
        ax3.plot(n_dims, ev, linestyle="--", color="gray", alpha=0.35,
                 label="Cum. explained variance")
        ax3.set_xlabel("Latent dimensions")
        ax3.set_ylabel("Score")
        ax3.set_title(f"{run_label} — {y_name} — Precision per class")
        ax3.set_ylim(0, 1.05)
        ax3.grid(True, alpha=0.3)
        ax3.legend()

    _set_xscale_log(ax1 if binary else ax3, n_dims)
    plt.tight_layout()
    outpath = RESULTS_FOLDER / f"{run_label}_{y_name}_classification.png"
    plt.savefig(outpath, dpi=200)
    plt.close()
    return outpath


def plot_linear_metrics(results: list[dict], y_name: str, run_label: str) -> Path:
    """
    Plot linear regression metrics (R², RMSE, MAE) vs number of latent dimensions.

    Parameters
    ----------
    results : list of dict
        Each dict from eval_linear, sorted by n_dims.
    y_name : str
    run_label : str

    Returns
    -------
    Path to saved figure.
    """
    n_dims = [r["n_dims"] for r in results]
    ev = [r["explained_variance"] for r in results]

    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(8, 10), sharex=True)

    ax1.plot(n_dims, [r["r2"] for r in results], marker="o", label="R²")
    ax1.plot(n_dims, ev, linestyle="--", color="gray", alpha=0.35,
             label="Cum. explained variance")
    ax1.set_ylabel("Score")
    ax1.set_title(f"{run_label} — {y_name} — R²")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(n_dims, [r["rmse"] for r in results], marker="o", label="RMSE")
    ax2.plot(n_dims, [r["mae"] for r in results], marker="o", label="MAE")
    ax2.plot(n_dims, ev, linestyle="--", color="gray", alpha=0.35,
             label="Cum. explained variance")
    ax2.set_xlabel("Latent dimensions")
    ax2.set_ylabel(y_name)
    ax2.set_title(f"{run_label} — {y_name} — Prediction errors")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    _set_xscale_log(ax2, n_dims)
    plt.tight_layout()
    outpath = RESULTS_FOLDER / f"{run_label}_{y_name}_linear.png"
    plt.savefig(outpath, dpi=200)
    plt.close()
    return outpath


def plot_confusion_matrix(result: dict, y_name: str, run_label: str) -> Path:
    """
    Plot a normalized confusion matrix from one result dict.

    Parameters
    ----------
    result : dict
        One dict from eval_logistic_binary or eval_logistic_multiclass.
    y_name : str
    run_label : str

    Returns
    -------
    Path to saved figure.
    """
    cm = result["confusion_matrix"].astype(float)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    classes = result["classes"]
    n_dims = result["n_dims"]

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=classes, yticklabels=classes, vmin=0, vmax=1)
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.title(f"{run_label} — {y_name} — Confusion matrix ({n_dims} dims)")
    plt.tight_layout()

    outpath = RESULTS_FOLDER / f"{run_label}_{y_name}_{n_dims}dims_confusion.png"
    plt.savefig(outpath, dpi=200)
    plt.close()
    return outpath


def plot_predicted_vs_true(result: dict, y_name: str, run_label: str) -> Path:
    """
    Scatter plot of predicted vs true values for linear regression.

    Parameters
    ----------
    result : dict
        One dict from eval_linear.
    y_name : str
    run_label : str

    Returns
    -------
    Path to saved figure.
    """
    Y_test = result["Y_test"]
    Y_pred = result["Y_pred"]
    n_dims = result["n_dims"]

    y_min = min(np.min(Y_test), np.min(Y_pred))
    y_max = max(np.max(Y_test), np.max(Y_pred))

    plt.figure(figsize=(6, 6))
    plt.scatter(Y_test, Y_pred, alpha=0.8)
    plt.plot([y_min, y_max], [y_min, y_max], linestyle="--", color="gray", alpha=0.7)
    plt.xlabel(f"True {y_name}")
    plt.ylabel(f"Predicted {y_name}")
    plt.title(f"{run_label} — {y_name} — Predicted vs True ({n_dims} dims)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    outpath = RESULTS_FOLDER / f"{run_label}_{y_name}_{n_dims}dims_pred_vs_true.png"
    plt.savefig(outpath, dpi=200)
    plt.close()
    return outpath


def plot_average_confusion_matrix(cms: list[np.ndarray], classes: list[str],
                                  title: str, outpath: Path) -> Path:
    """
    Average several row-normalized confusion matrices and plot the result.

    Parameters
    ----------
    cms : list of np.ndarray
        Raw (unnormalized) confusion matrices, all with same shape.
    classes : list of str
    title : str
    outpath : Path

    Returns
    -------
    outpath
    """
    cms_norm = [cm.astype(float) / cm.sum(axis=1, keepdims=True) for cm in cms]
    cm_avg = np.mean(cms_norm, axis=0)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_avg, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=classes, yticklabels=classes, vmin=0, vmax=1)
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    return outpath


# ── Internal helper ───────────────────────────────────────────────────────────

def _set_xscale_log(ax, n_dims: list) -> None:
    """Apply log x-scale with explicit tick labels when there are enough points."""
    if len(n_dims) > 3:
        ax.set_xscale("log")
        ax.set_xticks(n_dims)
        ax.set_xticklabels([str(x) for x in n_dims])
