# scripts/run_comparison_aepca.py
"""
Comparison plots: AE architectures and/or PCA spatial, metric vs latent dims.

Reads configs/comparison_aepca.yaml.
Output: one plot saved to results/ — no MLflow tracking.

Each series is identified by model_name + experiment_tag (+ optional params_filter).
  - model_name "PCA"  → reads from pca_spatial MLflow experiment (step = n_pcs)
  - any other name    → reads from autoencoder MLflow experiment (one run per latent_dim)

WARNINGs are printed (never blocking) when:
  - multiple runs match the same latent_dim for a series → most recent is used
  - key params differ within a series (potential data mix)
  - key params differ between series (potential apples-to-oranges comparison)

Usage:
    python scripts/run_comparison_aepca.py
"""

import yaml
import mlflow
import numpy as np
import pandas as pd
from pathlib import Path

from src.config import RESULTS_FOLDER
from src import tracking
from src.visualization.ae_plots import plot_comparison_curves

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "comparison_aepca.yaml"

_KEY_PARAMS = ["split_name", "n_train", "frame_tag", "image_roi_only"]


# ── MLflow query helpers ───────────────────────────────────────────────────────

def _build_filter(model_name: str, experiment_tag: str, params_filter: dict | None) -> str:
    parts = [f"params.experiment_tag = '{experiment_tag}'"]
    if model_name != "PCA":
        parts.append(f"params.model_name = '{model_name}'")
    for k, v in (params_filter or {}).items():
        parts.append(f"params.{k} = '{v}'")
    return " and ".join(parts)


def _warn_inconsistent_within(label: str, df: pd.DataFrame) -> None:
    for param in _KEY_PARAMS:
        col = f"params.{param}"
        if col not in df.columns:
            continue
        unique_vals = df[col].dropna().unique()
        if len(unique_vals) > 1:
            print(f"  WARNING series '{label}': inconsistent '{param}': {list(unique_vals)}")


def _warn_inconsistent_between(series_dfs: dict[str, pd.DataFrame]) -> None:
    for param in _KEY_PARAMS:
        col = f"params.{param}"
        vals_per_series = {}
        for label, df in series_dfs.items():
            if col not in df.columns:
                continue
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) == 1:
                vals_per_series[label] = unique_vals[0]
        all_vals = set(vals_per_series.values())
        if len(all_vals) > 1:
            detail = ", ".join(f"{k}={v}" for k, v in vals_per_series.items())
            print(f"  WARNING cross-series: '{param}' differs between series: {detail}")


# ── Data loaders ──────────────────────────────────────────────────────────────

def _load_ae_series(
    label: str,
    model_name: str,
    experiment_tag: str,
    params_filter: dict | None,
    metrics: list[str],
) -> dict[int, dict[str, float]]:
    """Return {latent_dim: {metric_key: value}} from autoencoder MLflow experiment."""
    filter_str = _build_filter(model_name, experiment_tag, params_filter)
    df = tracking.search_runs("autoencoder", filter_string=filter_str)

    if df.empty:
        print(f"  WARNING series '{label}': no runs found")
        return {}

    _warn_inconsistent_within(label, df)

    latdim_col = "params.latent_dimensions"
    if latdim_col not in df.columns:
        print(f"  WARNING series '{label}': 'latent_dimensions' param missing")
        return {}

    result: dict[int, dict[str, float]] = {}
    for latdim_str, group in df.groupby(latdim_col):
        try:
            latdim = int(latdim_str)
        except ValueError:
            continue
        if len(group) > 1:
            print(f"  WARNING series '{label}': {len(group)} runs for latdim={latdim}, using most recent")
        run = group.sort_values("start_time", ascending=False).iloc[0]
        metric_values = {}
        for m in metrics:
            col = f"metrics.{m}"
            if col in run.index and not pd.isna(run[col]):
                metric_values[m] = float(run[col])
        if metric_values:
            result[latdim] = metric_values

    return result


def _load_pca_series(
    label: str,
    experiment_tag: str,
    params_filter: dict | None,
    metrics: list[str],
) -> dict[int, dict[str, float]]:
    """Return {n_pcs: {metric_key: value}} from pca_spatial MLflow experiment."""
    filter_str = _build_filter("PCA", experiment_tag, params_filter)
    df = tracking.search_runs("pca_spatial", filter_string=filter_str)

    if df.empty:
        print(f"  WARNING series '{label}': no PCA runs found")
        return {}

    _warn_inconsistent_within(label, df)

    if len(df) > 1:
        print(f"  WARNING series '{label}': {len(df)} PCA runs found, using most recent")

    run_id = df.sort_values("start_time", ascending=False).iloc[0]["run_id"]
    client = mlflow.MlflowClient()

    result: dict[int, dict[str, float]] = {}
    for m in metrics:
        try:
            history = client.get_metric_history(run_id, m)
        except Exception:
            print(f"  WARNING series '{label}': metric '{m}' not found in PCA run {run_id[:8]}")
            continue
        for entry in history:
            n_pcs = entry.step
            if n_pcs not in result:
                result[n_pcs] = {}
            result[n_pcs][m] = entry.value

    return result


# ── Early stopping stats ───────────────────────────────────────────────────────

def _run_stats_nepochs(cfg: dict) -> None:
    stats_model = cfg["stats_model_name"]
    stats_tag = cfg["stats_experiment_tag"]
    stats_filter = cfg.get("stats_params_filter")

    filter_str = _build_filter(stats_model, stats_tag, stats_filter)
    df = tracking.search_runs("autoencoder", filter_string=filter_str)

    if df.empty:
        print(f"  No runs found for {stats_model}/{stats_tag}")
        return

    _warn_inconsistent_within(f"stats {stats_model}/{stats_tag}", df)

    stats: dict[int, dict] = {}
    for latdim_str, group in df.groupby("params.latent_dimensions"):
        try:
            latdim = int(latdim_str)
        except ValueError:
            continue
        epochs = [int(e) for e in group["params.best_epoch"].dropna()]
        if not epochs:
            continue
        stats[latdim] = {
            "epochs": epochs,
            "mean":   float(np.mean(epochs)),
            "median": float(np.median(epochs)),
            "std":    float(np.std(epochs)),
            "min":    int(np.min(epochs)),
            "max":    int(np.max(epochs)),
        }

    from src.training.ae_training import print_and_save_best_epochs_stats
    print_and_save_best_epochs_stats(stats, stats_model, stats_tag, RESULTS_FOLDER)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    metrics: list[str] = cfg["metrics"]
    xscale: str = cfg.get("xscale", "log")
    superpose_metrics: bool = cfg.get("superpose_metrics", False)
    series_cfg: list[dict] = cfg["series"]
    plotname: str | None = cfg.get("plotname")

    print("=== Loading MLflow data ===")
    all_series: list[dict] = []
    series_dfs: dict[str, pd.DataFrame] = {}

    for s in series_cfg:
        model_name = s["model_name"]
        experiment_tag = s["experiment_tag"]
        params_filter = s.get("params_filter")

        label = f"{model_name}/{experiment_tag}"
        if params_filter:
            label += "/" + "_".join(f"{k}={v}" for k, v in params_filter.items())

        print(f"\nSeries: {label}")

        if model_name == "PCA":
            data = _load_pca_series(label, experiment_tag, params_filter, metrics)
            df = tracking.search_runs("pca_spatial", filter_string=_build_filter("PCA", experiment_tag, params_filter))
        else:
            data = _load_ae_series(label, model_name, experiment_tag, params_filter, metrics)
            df = tracking.search_runs("autoencoder", filter_string=_build_filter(model_name, experiment_tag, params_filter))

        series_dfs[label] = df
        print(f"  → {len(data)} data points")
        all_series.append({"label": label, "data": data})

    print("\n=== Cross-series consistency check ===")
    _warn_inconsistent_between(series_dfs)

    print("\n=== Generating plot ===")
    if plotname:
        save_path = RESULTS_FOLDER / f"{plotname}.png"
    else:
        labels_str = "_vs_".join(
            s["model_name"] + "-" + s["experiment_tag"] for s in series_cfg
        )[:80]
        save_path = RESULTS_FOLDER / f"comparison_{labels_str}.png"

    plot_comparison_curves(
        series_data=all_series,
        metrics=metrics,
        xscale=xscale,
        superpose_metrics=superpose_metrics,
        save_path=save_path,
    )

    if cfg.get("run_stats_nepochs"):
        print("\n=== Early stopping stats ===")
        _run_stats_nepochs(cfg)


if __name__ == "__main__":
    main()
