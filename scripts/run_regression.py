# scripts/run_regression.py
"""
Regression pipeline on PCA or AE latent representations.

Source PCA  : loads a pre-trained PCA from a pca_spatial MLflow run and sweeps
              over cumvar_threshold_list → one regression per n_pc.
Source AE   : finds all autoencoder MLflow runs matching model_name /
              experiment_tag / split_name (+ optional params_filter), groups
              them by latent_dimensions, and runs one regression per latent_dim.

One MLflow run is created in the "regression" experiment per script execution.
Metrics are logged with step = n_dims (n_pc for PCA, latent_dim for AE).
Classifiers / regressors are saved as joblib artifacts inside that run.

Usage:
    python scripts/run_regression.py
"""

import re
import json
import yaml
import joblib
import numpy as np
import mlflow
import torch
from pathlib import Path

from src.config import RESULTS_FOLDER
from src.data import loader, splits as splt
from src.data.importdata import load_patient_metadata
from src.models import regression as reg
from src.models.ae_models import build_autoencoder
from src.training import ae_training as aet
from src.visualization import regression_plots as rgp
from src.visualization.ae_plots import collect_latent_vectors
from src import tracking

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "regression.yaml"

N_PATIENTS = 150


# ── Helpers ───────────────────────────────────────────────────────────────────

def _derive_split_name(special_split):
    return splt.DEFAULT_SPLIT_NAME if special_split is None else special_split


def _verify_split(run_id, expected: dict, client):
    """Raise if any expected param differs from what is stored in the MLflow run."""
    saved = client.get_run(run_id).data.params
    mismatches = [
        f"  {k}: stored={saved.get(k)!r}, expected={v!r}"
        for k, v in expected.items()
        if saved.get(k) != str(v)
    ]
    if mismatches:
        raise ValueError(
            f"Split / data mismatch with MLflow run {run_id}:\n" + "\n".join(mismatches)
        )


def _apply_split_to_Y(Y_full, n_train, n_test, special_split):
    """Return Y_train, Y_test using the same split logic as get_split_indices."""
    train_idx, _, test_idx, _ = splt.get_split_indices(
        n_train=n_train, n_val=0, n_test=n_test,
        special_split=special_split,
        n_patients=N_PATIENTS,
    )
    return Y_full[train_idx], Y_full[test_idx]


def _build_ae_filter(model_name, experiment_tag, split_name, params_filter):
    parts = [
        f"params.model_name = '{model_name}'",
        f"params.experiment_tag = '{experiment_tag}'",
        f"params.split_name = '{split_name}'",
    ]
    for k, v in (params_filter or {}).items():
        parts.append(f"params.{k} = '{v}'")
    return " and ".join(parts)


def _load_ae_model(run_id, model_name, latent_dim, dropout_rate, best_epoch, device, client):
    local_path = client.download_artifacts(run_id, f"model_{best_epoch}epochs.pth")
    model = build_autoencoder(model_name, latent_dim, dropout_rate=dropout_rate).to(device)
    model.load_state_dict(torch.load(local_path, map_location=device))
    model.eval()
    return model


def _encode_dataset(model, dataset, device):
    """Return latent vectors as float64 numpy array (n_samples, latent_dim)."""
    return collect_latent_vectors(model, dataset, device).astype(np.float64)


def _run_one_step(X_train, X_test, Y_train, Y_test,
                  n_dims, explained_variance, is_logistic, binary, logistic_C=1.0):
    """Standardize, fit, evaluate. Return (clf_or_reg, scaler, results_train, results_test)."""
    scaler = reg.fit_scaler(X_train)
    Xtr = scaler.transform(X_train).astype(np.float64)
    Xte = scaler.transform(X_test).astype(np.float64)

    if is_logistic:
        multi_class = not binary
        clf = reg.fit_logistic(Xtr, Y_train, multi_class=multi_class, C=logistic_C)
        eval_fn = reg.eval_logistic_binary if binary else reg.eval_logistic_multiclass
        r_train = eval_fn(clf, Xtr, Y_train, n_dims, explained_variance)
        r_test  = eval_fn(clf, Xte, Y_test,  n_dims, explained_variance)
        return clf, scaler, r_train, r_test
    else:
        model = reg.fit_linear(Xtr, Y_train)
        r_train = reg.eval_linear(model, Xtr, Y_train, n_dims, explained_variance)
        r_test  = reg.eval_linear(model, Xte, Y_test,  n_dims, explained_variance)
        return model, scaler, r_train, r_test


def _log_step_metrics(r_train, r_test, step, is_logistic, binary):
    if is_logistic:
        if binary:
            tracking.log_metric("accuracy_train",        r_train["accuracy"],  step=step)
            tracking.log_metric("roc_auc_train",         r_train["roc_auc"],   step=step)
            tracking.log_metric("precision_train",       r_train["precision"], step=step)
            tracking.log_metric("recall_train",          r_train["recall"],    step=step)
            tracking.log_metric("accuracy_test",         r_test["accuracy"],   step=step)
            tracking.log_metric("roc_auc_test",          r_test["roc_auc"],    step=step)
            tracking.log_metric("precision_test",        r_test["precision"],  step=step)
            tracking.log_metric("recall_test",           r_test["recall"],     step=step)
        else:
            tracking.log_metric("accuracy_train",        r_train["accuracy"],        step=step)
            tracking.log_metric("precision_macro_train", r_train["precision_macro"], step=step)
            tracking.log_metric("recall_macro_train",    r_train["recall_macro"],    step=step)
            tracking.log_metric("accuracy_test",         r_test["accuracy"],         step=step)
            tracking.log_metric("precision_macro_test",  r_test["precision_macro"],  step=step)
            tracking.log_metric("recall_macro_test",     r_test["recall_macro"],     step=step)
    else:
        tracking.log_metric("r2_train",   r_train["r2"],   step=step)
        tracking.log_metric("rmse_train", r_train["rmse"], step=step)
        tracking.log_metric("mae_train",  r_train["mae"],  step=step)
        tracking.log_metric("r2_test",    r_test["r2"],    step=step)
        tracking.log_metric("rmse_test",  r_test["rmse"],  step=step)
        tracking.log_metric("mae_test",   r_test["mae"],   step=step)


# ── PCA source ────────────────────────────────────────────────────────────────

def _run_pca_source(cfg, Y_full, client):
    n_train    = cfg["n_train"]
    n_test     = N_PATIENTS - n_train
    special_split = cfg.get("special_split")
    pca_run_id = cfg["pca_run_id"]
    y_name     = cfg["y_name"]
    binary     = cfg.get("group_binary", False) and y_name == "group"
    is_logistic = y_name == "group"

    # ── Read params from PCA run once (avoid redundant API calls) ────────────
    pca_params      = client.get_run(pca_run_id).data.params
    frame_tag       = pca_params.get("frame_tag", "ED")
    use_both_frames = frame_tag == "ED+ES"
    frame_type      = "ED" if use_both_frames else frame_tag

    # ── Load X ───────────────────────────────────────────────────────────────
    print("Loading image data...")
    X_train, _, X_test, split_name = loader.load_numpy_splits(
        source_folder=pca_params["source_folder"],
        cache_folder=pca_params.get("cache_folder", "X_vectors"),
        n_train=n_train, n_val=0, n_test=n_test,
        special_split=special_split,
        use_both_frames=use_both_frames,
        frame_type=frame_type,
        image_roi_only=pca_params.get("image_roi_only", "True") == "True",
        recalculate=False,
    )
    print(f"Split: {split_name} | train={len(X_train)} | test={len(X_test)}")

    # ── Verify split against PCA run ─────────────────────────────────────────
    _verify_split(pca_run_id, {"split_name": split_name, "n_train": n_train}, client)
    print(f"Split verified against PCA run {pca_run_id} ✓")

    # ── Load PCA ──────────────────────────────────────────────────────────────
    pca_filename = f"pca_{frame_tag}.joblib"
    local_path   = client.download_artifacts(pca_run_id, pca_filename)
    pca          = joblib.load(local_path)
    print(f"PCA loaded ({pca.n_components_} components) from run {pca_run_id}")

    # ── Row-centre X (same preprocessing as run_pca_spatial.py) ──────────────
    row_means_train = X_train.mean(axis=1, keepdims=True)
    row_means_test  = X_test.mean(axis=1, keepdims=True)
    X_train_pca = pca.transform(X_train - row_means_train)
    X_test_pca  = pca.transform(X_test  - row_means_test)

    # ── Y (duplicate if both frames) ─────────────────────────────────────────
    Y_train, Y_test = _apply_split_to_Y(Y_full, n_train, n_test, special_split)
    if use_both_frames:
        Y_train = np.concatenate([Y_train, Y_train])
        Y_test  = np.concatenate([Y_test,  Y_test])
    if binary:
        bin_val = cfg["group_bin_value"]
        Y_train = (Y_train == bin_val).astype(int)
        Y_test  = (Y_test  == bin_val).astype(int)

    logistic_C = cfg.get("logistic_C") or (0.5 if use_both_frames else 1.0)

    # ── Sweep over cumvar thresholds (deduplicate n_pc) ──────────────────────
    thresholds     = cfg["cumvar_threshold_list"]
    n_pc_confusion = cfg.get("n_pc_confusion", 12)
    run_label      = cfg.get("experiment_tag", "baseline")

    # Map each threshold to its n_pc; skip duplicates (keep first occurrence)
    seen_n_pc: set[int] = set()
    threshold_n_pc_pairs: list[tuple[float, int]] = []
    for threshold in thresholds:
        n_pc = reg.n_pc_for_variance(pca, threshold)
        if n_pc not in seen_n_pc:
            seen_n_pc.add(n_pc)
            threshold_n_pc_pairs.append((threshold, n_pc))

    results_test_all  = []
    results_train_all = []

    with tracking.start_run("regression", _run_name(cfg, split_name)):
        tracking.log_params(_build_params(cfg, split_name))
        tracking.log_artifact(CONFIG_PATH)

        for threshold, n_pc in threshold_n_pc_pairs:
            cumvar = float(np.sum(pca.explained_variance_ratio_[:n_pc]))

            Xtr = X_train_pca[:, :n_pc]
            Xte = X_test_pca[:,  :n_pc]

            _, _, r_train, r_test = _run_one_step(
                Xtr, Xte, Y_train, Y_test,
                n_pc, cumvar, is_logistic, binary, logistic_C,
            )
            _log_step_metrics(r_train, r_test, step=n_pc, is_logistic=is_logistic, binary=binary)
            _save_result_json(r_train, r_test, n_pc, label="pc")

            results_train_all.append(r_train)
            results_test_all.append(r_test)
            print(f"  n_pc={n_pc:4d} | cumvar={cumvar:.3f} | "
                  + _result_summary(r_test, is_logistic, binary))

        results_train_all.sort(key=lambda r: r["n_dims"])
        results_test_all.sort(key=lambda r: r["n_dims"])

        # ── Plots ─────────────────────────────────────────────────────────────
        _save_plots(results_train_all, results_test_all,
                    n_pc_confusion, y_name, run_label, is_logistic, binary)


# ── AE source ─────────────────────────────────────────────────────────────────

def _run_ae_source(cfg, Y_full, client):
    n_train       = cfg["n_train"]
    n_test        = N_PATIENTS - n_train
    special_split = cfg.get("special_split")
    split_name    = _derive_split_name(special_split)
    y_name        = cfg["y_name"]
    binary        = cfg.get("group_binary", False) and y_name == "group"
    is_logistic   = y_name == "group"
    ae_cfg        = cfg["ae_source"]

    # ── Search AE runs ────────────────────────────────────────────────────────
    filter_str = _build_ae_filter(
        ae_cfg["model_name"], ae_cfg["experiment_tag"],
        ae_cfg.get("split_name", split_name),
        ae_cfg.get("params_filter"),
    )
    df = tracking.search_runs("autoencoder", filter_string=filter_str)
    if df.empty:
        raise ValueError(f"No autoencoder runs found matching: {filter_str}")

    # Group by latent_dimensions; warn if ambiguous
    runs_by_latdim = {}
    for _, row in df.iterrows():
        latdim = int(row["params.latent_dimensions"])
        runs_by_latdim.setdefault(latdim, []).append(row)

    for latdim, rows in runs_by_latdim.items():
        if len(rows) > 1:
            print(f"WARNING: {len(rows)} runs found for latent_dim={latdim} — "
                  f"using most recent: {rows[0]['run_id']}")

    sorted_latdims = sorted(runs_by_latdim.keys())
    print(f"Found {len(sorted_latdims)} latent_dim(s): {sorted_latdims}")

    # ── Read data loading params from the first AE run ───────────────────────
    ref_params     = runs_by_latdim[sorted_latdims[0]][0]
    frame_tag      = ref_params.get("params.frame_tag", "ED")
    use_both       = frame_tag == "ED+ES"
    frame_type     = "ED" if use_both else frame_tag
    image_roi_only = ref_params.get("params.image_roi_only", "True") == "True"
    source_folder  = ref_params.get("params.source_folder", "registered_frames")

    # Verify all AE runs share the same split
    for latdim in sorted_latdims:
        ae_run_id = runs_by_latdim[latdim][0]["run_id"]
        _verify_split(ae_run_id, {"split_name": split_name, "n_train": n_train}, client)
    print(f"Split verified against all {len(sorted_latdims)} AE runs ✓")

    # ── Load tensor datasets (same split as AE training) ─────────────────────
    print("Loading image data (tensor format for AE encoding)...")
    train_ds, _, test_ds, _, _ = loader.load_tensor_datasets(
        source_folder=source_folder,
        cache_folder="X_vectors",
        n_train=n_train, n_val=0, n_test=n_test,
        special_split=special_split,
        use_both_frames=use_both,
        frame_type=frame_type,
        image_roi_only=image_roi_only,
        recalculate=False,
    )
    print(f"Split: {split_name} | train={len(train_ds)} | test={len(test_ds)}")

    # ── Y (duplicate if both frames) ─────────────────────────────────────────
    Y_train_base, Y_test_base = _apply_split_to_Y(Y_full, n_train, n_test, special_split)
    if use_both:
        Y_train_base = np.concatenate([Y_train_base, Y_train_base])
        Y_test_base  = np.concatenate([Y_test_base,  Y_test_base])
    if binary:
        bin_val = cfg["group_bin_value"]
        Y_train_base = (Y_train_base == bin_val).astype(int)
        Y_test_base  = (Y_test_base  == bin_val).astype(int)

    logistic_C = cfg.get("logistic_C") or (0.5 if use_both else 1.0)
    device = aet.get_device()
    n_pc_confusion = cfg.get("n_pc_confusion", sorted_latdims[len(sorted_latdims) // 2])
    run_label = cfg.get("experiment_tag", "baseline")

    results_train_all = []
    results_test_all  = []

    with tracking.start_run("regression", _run_name(cfg, split_name)):
        tracking.log_params(_build_params(cfg, split_name))
        tracking.log_artifact(CONFIG_PATH)

        for latdim in sorted_latdims:
            ae_run_id  = runs_by_latdim[latdim][0]["run_id"]
            ae_row     = runs_by_latdim[latdim][0]
            model_name = ae_row["params.model_name"]
            dropout    = float(ae_row.get("params.dropout_rate", 0.0))
            best_epoch = int(
                ae_row.get("params.best_epoch") or ae_row.get("params.n_epochs", 100)
            )

            model = _load_ae_model(ae_run_id, model_name, latdim, dropout, best_epoch, device, client)
            Z_train = _encode_dataset(model, train_ds, device)
            Z_test  = _encode_dataset(model, test_ds,  device)
            print(f"  latent_dim={latdim} | encoded train={Z_train.shape} test={Z_test.shape}")

            _, _, r_train, r_test = _run_one_step(
                Z_train, Z_test, Y_train_base, Y_test_base,
                latdim, 1.0, is_logistic, binary, logistic_C,
            )
            _log_step_metrics(r_train, r_test, step=latdim, is_logistic=is_logistic, binary=binary)
            _save_result_json(r_train, r_test, latdim, label="latdim")

            results_train_all.append(r_train)
            results_test_all.append(r_test)
            print(f"  latent_dim={latdim:3d} | " + _result_summary(r_test, is_logistic, binary))

        # ── Plots ─────────────────────────────────────────────────────────────
        _save_plots(results_train_all, results_test_all,
                    n_pc_confusion, y_name, run_label, is_logistic, binary)


# ── Result JSON helpers (committed to git via mlruns/) ────────────────────────

def _result_to_serializable(r: dict) -> dict:
    """Convert numpy arrays in a result dict to JSON-serializable types."""
    out = {}
    for k, v in r.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, dict):
            out[k] = {str(kk): float(vv) for kk, vv in v.items()}
        else:
            out[k] = v
    return out


def _save_result_json(r_train: dict, r_test: dict, n_dims: int, label: str):
    """Save train+test result dicts as a JSON artifact in the active MLflow run."""
    stage = RESULTS_FOLDER / "tmp_artifacts"
    stage.mkdir(parents=True, exist_ok=True)
    path = stage / f"results_{n_dims}{label}.json"
    with open(path, "w") as f:
        json.dump({"train": _result_to_serializable(r_train),
                   "test":  _result_to_serializable(r_test)}, f)
    tracking.log_artifact(path)


def _load_result_json(run_id: str, n_dims: int, label: str, client) -> tuple[dict, dict]:
    """Download and deserialize a result JSON artifact; restore numpy arrays."""
    local_path = client.download_artifacts(run_id, f"results_{n_dims}{label}.json")
    with open(local_path) as f:
        data = json.load(f)
    for r in [data["train"], data["test"]]:
        if "confusion_matrix" in r:
            r["confusion_matrix"] = np.array(r["confusion_matrix"])
        if "Y_pred" in r:
            r["Y_pred"] = np.array(r["Y_pred"])
        if "Y_test" in r:
            r["Y_test"] = np.array(r["Y_test"])
    return data["train"], data["test"]


def _list_result_dims(run_id: str, label: str, client) -> list[int]:
    """Return sorted list of n_dims for which result JSON artifacts exist."""
    artifacts = client.list_artifacts(run_id)
    dims = []
    for a in artifacts:
        m = re.match(rf"results_(\d+){re.escape(label)}\.json", Path(a.path).name)
        if m:
            dims.append(int(m.group(1)))
    return sorted(dims)


# ── Shared plot helper ────────────────────────────────────────────────────────

def _save_plots(results_train, results_test,
                n_dims_confusion, y_name, run_label, is_logistic, binary):
    available = [r["n_dims"] for r in results_test]

    if is_logistic:
        p_train = rgp.plot_logistic_metrics(results_train, y_name, run_label + "_train", binary)
        p_test  = rgp.plot_logistic_metrics(results_test,  y_name, run_label + "_test",  binary)
        tracking.log_artifact(p_train)
        tracking.log_artifact(p_test)

        cm_dims   = min(available, key=lambda x: abs(x - n_dims_confusion))
        cm_result = next(r for r in results_test if r["n_dims"] == cm_dims)
        p_cm = rgp.plot_confusion_matrix(cm_result, y_name, run_label)
        tracking.log_artifact(p_cm)
    else:
        p_train = rgp.plot_linear_metrics(results_train, y_name, run_label + "_train")
        p_test  = rgp.plot_linear_metrics(results_test,  y_name, run_label + "_test")
        tracking.log_artifact(p_train)
        tracking.log_artifact(p_test)

        pvt_dims   = min(available, key=lambda x: abs(x - n_dims_confusion))
        pvt_result = next(r for r in results_test if r["n_dims"] == pvt_dims)
        p_pvt = rgp.plot_predicted_vs_true(pvt_result, y_name, run_label)
        tracking.log_artifact(p_pvt)


def _result_summary(r, is_logistic, binary):
    if is_logistic:
        if binary:
            return f"acc={r['accuracy']:.3f}  auc={r['roc_auc']:.3f}"
        return f"acc={r['accuracy']:.3f}  prec_macro={r['precision_macro']:.3f}"
    return f"r2={r['r2']:.3f}  rmse={r['rmse']:.2f}"


def _run_name(cfg, split_name):
    src = cfg["source_type"]
    tag = cfg.get("experiment_tag", "baseline")
    y   = cfg["y_name"]
    return f"regression_{src}_{cfg['n_train']}patients_{split_name}_{y}_{tag}"


def _build_params(cfg, split_name):
    params = {
        "source_type":    cfg["source_type"],
        "y_name":         cfg["y_name"],
        "group_binary":   str(cfg.get("group_binary", False)),
        "group_bin_value": cfg.get("group_bin_value", ""),
        "n_train":        cfg["n_train"],
        "n_test":         N_PATIENTS - cfg["n_train"],
        "split_name":     split_name,
        "experiment_tag": cfg.get("experiment_tag", "baseline"),
        "n_pc_confusion": cfg.get("n_pc_confusion", 12),
    }
    if cfg["source_type"] == "pca":
        params["pca_run_id"] = cfg["pca_run_id"]
    else:
        ae = cfg["ae_source"]
        params["ae_model_name"]      = ae["model_name"]
        params["ae_experiment_tag"]  = ae["experiment_tag"]
        params["ae_split_name"]      = ae.get("split_name", split_name)
    return params


# ── Plot-only mode ────────────────────────────────────────────────────────────

def _plot_only(cfg, client):
    """Reload result JSON artifacts from a previous run and regenerate plots."""
    load_run_id  = cfg["load_run_id"]
    run_params   = client.get_run(load_run_id).data.params

    source_type  = run_params["source_type"]
    y_name       = run_params["y_name"]
    binary       = run_params["group_binary"] == "True"
    is_logistic  = y_name == "group"
    run_label    = run_params.get("experiment_tag", "baseline")
    n_pc_confusion = int(run_params.get("n_pc_confusion", 12))

    label = "pc" if source_type == "pca" else "latdim"
    dims  = _list_result_dims(load_run_id, label, client)
    if not dims:
        raise ValueError(
            f"No result JSON artifacts found in run {load_run_id}. "
            "Run in CALC mode first."
        )

    print(f"plot_only | run={load_run_id} | source={source_type} | "
          f"y={y_name} | dims={dims}")

    results_train_all, results_test_all = [], []
    for n_dims in dims:
        r_train, r_test = _load_result_json(load_run_id, n_dims, label, client)
        results_train_all.append(r_train)
        results_test_all.append(r_test)

    with tracking.resume_run(load_run_id):
        _save_plots(results_train_all, results_test_all,
                    n_pc_confusion, y_name, run_label, is_logistic, binary)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    tracking._setup()
    client = mlflow.MlflowClient()

    y_name = cfg["y_name"]
    n_train = cfg["n_train"]
    n_test  = N_PATIENTS - n_train
    special_split = cfg.get("special_split")
    split_name = _derive_split_name(special_split)

    if cfg.get("plot_only"):
        if not cfg.get("load_run_id"):
            raise ValueError("plot_only requires load_run_id in regression.yaml")
        _plot_only(cfg, client)
        return

    print(f"Regression | source={cfg['source_type']} | y={y_name} | "
          f"n_train={n_train} | split={split_name}")

    # ── Load Y (all patients, split applied inside each source function) ──────
    Y_full = load_patient_metadata(y_name, N_PATIENTS)
    print(f"Y loaded: {y_name}, {len(Y_full)} patients")

    if cfg["source_type"] == "pca":
        if not cfg.get("pca_run_id"):
            raise ValueError("pca_run_id must be set in regression.yaml for source_type='pca'")
        _run_pca_source(cfg, Y_full, client)

    elif cfg["source_type"] == "ae":
        if not cfg.get("ae_source"):
            raise ValueError("ae_source must be set in regression.yaml for source_type='ae'")
        _run_ae_source(cfg, Y_full, client)

    else:
        raise ValueError(f"Unknown source_type: {cfg['source_type']!r}. Use 'pca' or 'ae'.")


if __name__ == "__main__":
    main()
