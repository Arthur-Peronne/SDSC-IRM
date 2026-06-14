# src/visualization/pca_plots.py
"""
PCA visualization functions — shared by the temporal and spatial pipelines.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.config import RESULTS_FOLDER
from src.visualization.ae_plots import ae_plotcompare_onepatient


# ── Explained variance ────────────────────────────────────────────────────────

def plot_pca_explipower(pca, patient_name):
    n_components = len(pca.explained_variance_ratio_)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
    fig.suptitle(patient_name + ': variance explained by principal components')
    ax1.plot(pca.explained_variance_ratio_, marker='o', linestyle='--')
    ax1.set_ylabel('Explained variance')
    ax1.set_ylim(0, max(pca.explained_variance_ratio_) * 1.1)
    ax1.set_xticks(range(0, n_components, max(1, int(n_components / 30))))
    ax1.set_xticklabels(range(1, n_components + 1, max(1, int(n_components / 30))))
    ax1.grid(True)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    ax2.plot(cumulative_variance, marker='o', linestyle='--')
    ax2.set_xticks(range(0, n_components, max(1, int(n_components / 30))))
    ax2.set_xticklabels(range(1, n_components + 1, max(1, int(n_components / 30))))
    ax2.set_xlabel('Number of principal components')
    ax2.set_ylabel('Cumulative explained variance')
    ax2.set_ylim(0, 1.05)
    ax2.grid(True)
    plt.savefig(RESULTS_FOLDER / f"{patient_name}_PCA_explainedvariance.png")
    plt.close()


# ── PC values in eigenbase ────────────────────────────────────────────────────

def plot_pcvalues_2d(X_reduced, pc_n1, pc_n2, patient_str, details_str,
                     scale_str='Time (Epoch)', segments=True, axisscale_fixed=True):
    fig, ax1 = plt.subplots(1, 1)
    colors = plt.cm.coolwarm(np.linspace(0, 1, X_reduced.shape[0]))
    scatter1 = ax1.scatter(
        X_reduced[:, pc_n1], X_reduced[:, pc_n2],
        s=40, c=np.linspace(0, 1, X_reduced.shape[0]), cmap='coolwarm',
    )
    if segments:
        for i in range(X_reduced.shape[0] - 1):
            ax1.plot(X_reduced[i:i+2, pc_n1], X_reduced[i:i+2, pc_n2],
                     color=colors[i], linestyle='-', linewidth=1)
    cbar = plt.colorbar(scatter1, ax=ax1)
    cbar.set_label(scale_str)
    cbar.set_ticks(np.linspace(0, 1, 6))
    cbar.set_ticklabels([f'{i+1}' for i in np.linspace(0, X_reduced.shape[0]-1, 6, dtype=int)])
    ax1.set(
        title=f"{patient_str} : Principal Components {pc_n1+1} and {pc_n2+1}",
        xlabel=f"Principal Component {pc_n1+1}",
        ylabel=f"Principal Component {pc_n2+1}",
    )
    axis1, axis2 = (0, 1) if axisscale_fixed else (pc_n1, pc_n2)
    x_ticks = np.linspace(-max(abs(X_reduced[:, axis1].min()), X_reduced[:, axis1].max()),
                           max(abs(X_reduced[:, axis1].min()), X_reduced[:, axis1].max()), 7)
    y_ticks = np.linspace(-max(abs(X_reduced[:, axis2].min()), X_reduced[:, axis2].max()),
                           max(abs(X_reduced[:, axis2].min()), X_reduced[:, axis2].max()), 7)
    ax1.set_xticks(x_ticks)
    ax1.set_yticks(y_ticks)
    plt.savefig(RESULTS_FOLDER / f"{patient_str}{details_str}_{pc_n1+1}and{pc_n2+1}.png")
    plt.close()


# ── PC values colored by patient metadata ─────────────────────────────────────

def plot_pcvalues_2d_meta(X_reduced, pc_n1, pc_n2, metainfo_str, metainfo_list,
                          axisscale_fixed=True, extremes_toremove=15) -> Path:
    fig, ax1 = plt.subplots(1, 1)
    botlimit = sorted(metainfo_list)[extremes_toremove]
    toplimit = sorted(metainfo_list)[-extremes_toremove]
    colors = [max(min((v - botlimit) / (toplimit - botlimit), 1), 0) for v in metainfo_list]
    scatter1 = ax1.scatter(X_reduced[:, pc_n1], X_reduced[:, pc_n2],
                           s=40, c=colors, cmap='coolwarm')
    cbar = plt.colorbar(scatter1, ax=ax1)
    cbar.set_label(metainfo_str)
    cbar.set_ticks(np.linspace(0, 1, 6))
    cbar.set_ticklabels([f'{i}' for i in np.linspace(botlimit, toplimit, 6)])
    ax1.set(
        title=f"Principal Components {pc_n1+1} and {pc_n2+1} — {metainfo_str}",
        xlabel=f"Principal Component {pc_n1+1}",
        ylabel=f"Principal Component {pc_n2+1}",
    )
    axis1, axis2 = (0, 1) if axisscale_fixed else (pc_n1, pc_n2)
    max_x = max(abs(X_reduced[:, axis1].min()), abs(X_reduced[:, axis1].max()))
    max_y = max(abs(X_reduced[:, axis2].min()), abs(X_reduced[:, axis2].max()))
    ax1.set_xticks(np.linspace(-max_x, max_x, 7))
    ax1.set_yticks(np.linspace(-max_y, max_y, 7))
    outpath = RESULTS_FOLDER / f"pc_allpatientsepoch0_{metainfo_str}_{pc_n1+1}and{pc_n2+1}.png"
    plt.savefig(outpath)
    plt.close()
    return outpath


def plot_pcvalues_2d_metacat(X_reduced, pc_n1, pc_n2, metainfo_str, metainfo_list,
                             axisscale_fixed=True) -> Path:
    fig, ax1 = plt.subplots(1, 1)
    groups_unique = sorted(set(metainfo_list))
    cmap = plt.get_cmap("tab10")
    color_dict = {g: cmap(i % 10) for i, g in enumerate(groups_unique)}
    for g in groups_unique:
        indices = [i for i, grp in enumerate(metainfo_list) if grp == g]
        ax1.scatter(X_reduced[indices, pc_n1], X_reduced[indices, pc_n2],
                    s=40, color=color_dict[g], label=g)
    ax1.set(
        title=f"Principal Components {pc_n1+1} and {pc_n2+1} — {metainfo_str}",
        xlabel=f"Principal Component {pc_n1+1}",
        ylabel=f"Principal Component {pc_n2+1}",
    )
    ax1.legend(title=metainfo_str)
    axis1, axis2 = (0, 1) if axisscale_fixed else (pc_n1, pc_n2)
    max_x = max(abs(X_reduced[:, axis1].min()), abs(X_reduced[:, axis1].max()))
    max_y = max(abs(X_reduced[:, axis2].min()), abs(X_reduced[:, axis2].max()))
    ax1.set_xticks(np.linspace(-max_x, max_x, 7))
    ax1.set_yticks(np.linspace(-max_y, max_y, 7))
    outpath = RESULTS_FOLDER / f"pc_allpatientsepoch0_{metainfo_str}_{pc_n1+1}and{pc_n2+1}.png"
    plt.savefig(outpath)
    plt.close()
    return outpath


# ── Reconstruction comparison (selected patients) ─────────────────────────────

def pca_plotcompare_selected(
    patients_torecons,
    X_train_pca,
    X_val_pca,
    X_test_pca,
    pca,
    latent_dimensions,
    original_shape,
    use_both_frames,
    n_development,
    n_train_images,
    n_val_images,
    split_name,
    row_means_train,
    row_means_val,
    row_means_test,
):
    """
    Plot PCA reconstruction for selected patients.

    row_means_{train/val/test} are the per-patient means subtracted before PCA,
    needed to correctly de-center the reconstruction.
    Shape: (n_images_in_split, 1)
    """
    n_train_real = n_train_images // 2 if use_both_frames else n_train_images

    from src.config import RESULTS_FOLDER
    saved_paths = []

    for real_patient, frame_type in patients_torecons:
        if real_patient <= n_train_real:
            X_pca_sub = X_train_pca
            row_means = row_means_train
            pos = real_patient - 1
        elif real_patient <= n_development:
            X_pca_sub = X_val_pca
            row_means = row_means_val
            pos = real_patient - n_train_real - 1
        else:
            X_pca_sub = X_test_pca
            row_means = row_means_test
            pos = real_patient - n_development - 1

        if use_both_frames and frame_type == "ES":
            pos += len(X_pca_sub) // 2

        if pos < 0 or pos >= len(X_pca_sub):
            raise IndexError(f"Invalid index {pos} for patient {real_patient} ({frame_type})")

        patient_row_mean = float(row_means[pos, 0])
        x_recon_flat = (
            X_pca_sub[pos, :latent_dimensions]
            @ pca.components_[:latent_dimensions, :]
            + pca.mean_
            + patient_row_mean
        )

        pat_str = f"patient{real_patient}"
        ae_str  = f"PCA_{split_name}_{latent_dimensions}dims"

        ae_plotcompare_onepatient(
            x_recon_flat.reshape(original_shape),
            real_patient,
            "registered_frames",
            ae_str=ae_str,
            pat_str=pat_str,
            details_str=frame_type,
            frame_type=frame_type,
        )
        saved_paths += [
            RESULTS_FOLDER / f"{pat_str}_original_{frame_type}.png",
            RESULTS_FOLDER / f"{pat_str}_{ae_str}_{frame_type}.png",
        ]

    return saved_paths



# ── Metrics vs latent dimensions ──────────────────────────────────────────────

def plot_pca_metrics_vs_latentdim(
    run_id,
    datasets=None,
    metrics=None,
    xscale="linear",
    plot_std=True,
    title=None,
    save_path=None,
):
    """
    Plot reconstruction metrics vs number of PCA components.

    Reads step-based metrics from MLflow (step = number of PCA components).
    Deduplicates by keeping the most recent value per step, so re-running
    compute_metrics safely replaces old values in the plot.

    Parameters
    ----------
    run_id : str
        MLflow run ID containing the step-based metrics.
    datasets : list of str, optional
        Datasets to plot. Default: ["train", "test"]. Add "validation" if needed.
    metrics : list of str, optional
        Metric names. Default: ["R2", "MSE"].
    xscale : str
        "linear" or "log".
    plot_std : bool
        If True, draw a shaded ±1 std band around the mean curve.
    title : str or None
        Optional figure title (e.g. the MLflow run_name).
    save_path : Path or str or None
        Output PDF path. Defaults to RESULTS_FOLDER/pca_metrics_vs_latentdim_{run_id[:8]}.pdf.

    Returns
    -------
    Path — the saved file path.
    """
    import mlflow

    if datasets is None:
        datasets = ["train", "test"]
    if metrics is None:
        metrics = ["R2", "MSE"]

    COLORS = {"train": "steelblue", "validation": "darkorange", "test": "seagreen"}
    LABELS = {"train": "Train", "validation": "Validation", "test": "Test"}

    client = mlflow.MlflowClient()

    def _dedup(hist):
        latest = {}
        for m in hist:
            if m.step not in latest or m.timestamp > latest[m.step].timestamp:
                latest[m.step] = m
        return dict(sorted(latest.items()))

    # ── Load from MLflow ──────────────────────────────────────────────────────
    data = {}
    for dataset in datasets:
        for metric in metrics:
            mean_hist = client.get_metric_history(run_id, f"{dataset}_{metric}_mean")
            std_hist  = client.get_metric_history(run_id, f"{dataset}_{metric}_std")
            if not mean_hist:
                continue
            mean_by_step = _dedup(mean_hist)
            std_by_step  = _dedup(std_hist) if std_hist else {}
            dims  = np.array(list(mean_by_step.keys()))
            means = np.array([mean_by_step[d].value for d in dims])
            stds  = np.array([std_by_step[d].value if d in std_by_step else 0.0 for d in dims])
            data.setdefault(dataset, {})[metric] = (dims, means, stds)

    if not data:
        raise ValueError(
            f"No metric data found for run_id={run_id!r}. "
            "Run with compute_metrics: true first."
        )

    # ── Plot ──────────────────────────────────────────────────────────────────
    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 4 * n_metrics), sharex=True)
    if n_metrics == 1:
        axes = [axes]
    if title:
        fig.suptitle(title, fontsize=11)

    for ax, metric in zip(axes, metrics):
        for dataset in datasets:
            if dataset not in data or metric not in data[dataset]:
                continue
            dims, means, stds = data[dataset][metric]
            color = COLORS.get(dataset, "gray")
            ax.plot(dims, means, color=color,
                    label=LABELS.get(dataset, dataset), linewidth=1.5)
            if plot_std:
                lower = np.maximum(means - stds, 0) if metric in {"MSE", "RMSE", "MAE"} else means - stds
                ax.fill_between(dims, lower, means + stds, alpha=0.2, color=color)

        ax.set_ylabel(metric)
        ax.set_xscale(xscale)
        ax.grid(True, alpha=0.3)
        ax.legend()
        if metric == "R2":
            ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)

    axes[-1].set_xlabel("Number of PCA components")
    plt.tight_layout()

    if save_path is None:
        save_path = RESULTS_FOLDER / f"pca_metrics_vs_latentdim_{run_id[:8]}.png"
    save_path = Path(save_path)
    plt.savefig(save_path)
    plt.close()
    return save_path
