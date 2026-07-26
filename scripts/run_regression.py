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

classifier_type (set in configs/regression.yaml):
    "logistic"      → LogisticRegression (default, existing behaviour)
    "random_forest" → RandomForestClassifier
    "xgboost"       → XGBClassifier

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
import argparse

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

def _parse_args():
    p = argparse.ArgumentParser(
        description="Encode AE/PCA latents and evaluate a downstream classifier/regressor."
    )
    p.add_argument("--trial-id", default=None,
                   help="Tag the MLflow run of this invocation with trial_id=<id> "
                        "(used by ai_agent/driver.py to read back exactly its runs).")
    p.add_argument("--ae-trial-tag", default=None,
                   help="[not yet used] Trouver le run AE par tags.trial_id=<tag> "
                        "au lieu des filtres model_name/experiment_tag/split_name habituels.")
    p.add_argument("--ae-filter", action="append", default=[], metavar="KEY=VALUE",
                   help="[not yet used] Condition(s) params.KEY='VALUE' supplémentaire(s) "
                        "pour désambiguïser plusieurs runs AE partageant le même --ae-trial-tag.")
    return p.parse_args()
    
def _derive_split_name(special_split):
    return splt.DEFAULT_SPLIT_NAME if special_split is None else special_split


def _verify_split(run_id, expected: dict, client):
    """Raise if any expected param differs from what is stored in the MLflow run.
    Params absent from the stored run (e.g. older runs predating this param)
    are skipped rather than treated as a mismatch.
    """
    saved = client.get_run(run_id).data.params
    mismatches = [
        f"  {k}: stored={saved.get(k)!r}, expected={v!r}"
        for k, v in expected.items()
        if k in saved and saved.get(k) != str(v)
    ]
    if mismatches:
        raise ValueError(
            f"Split / data mismatch with MLflow run {run_id}:\n" + "\n".join(mismatches)
        )

def _apply_split_to_Y(Y_full, n_train, n_val, n_test, special_split, stratify_ongroup=False, 
                                                 recalculate_y=False, y_cache_folder="Y_vectors"):
    """Return Y_train, Y_val, Y_test using the same split logic as get_split_indices.
    Y_val is an empty array when n_val=0 (unchanged behaviour for existing callers)."""
    strat = load_patient_metadata("group", N_PATIENTS, recalculate=recalculate_y, cache_folder=y_cache_folder) if stratify_ongroup else None
    train_idx, val_idx, test_idx, _ = splt.get_split_indices(
        n_train=n_train, n_val=n_val, n_test=n_test,
        special_split=special_split,
        stratify=strat,
        n_patients=N_PATIENTS,
    )
    return Y_full[train_idx], Y_full[val_idx], Y_full[test_idx]

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


def _get_classifier_kwargs(cfg, use_both_frames: bool = False) -> dict:
    """
    Build the classifier-specific kwargs dict to unpack into _run_one_step().

    Parameters
    ----------
    cfg : dict
        Full YAML config.
    use_both_frames : bool
        If True and classifier is logistic and logistic_C is not set explicitly,
        C is halved (0.5) to compensate for correlated ED+ES pairs.
    """
    classifier_type = cfg.get("classifier_type", "logistic")
    kwargs = {"classifier_type": classifier_type}

    if classifier_type == "logistic":
        # Explicit logistic_C in YAML takes priority; otherwise auto based on frames
        kwargs["logistic_C"] = cfg.get("logistic_C") or (0.5 if use_both_frames else 1.0)

    elif classifier_type == "random_forest":
        rf = cfg.get("random_forest") or {}
        kwargs["rf_n_estimators"]     = rf.get("n_estimators", 300)
        kwargs["rf_max_depth"]        = rf.get("max_depth", None)
        kwargs["rf_min_samples_leaf"] = rf.get("min_samples_leaf", 1)

    elif classifier_type == "xgboost":
        xgb = cfg.get("xgboost") or {}
        kwargs["xgb_n_estimators"]     = xgb.get("n_estimators", 300)
        kwargs["xgb_max_depth"]        = xgb.get("max_depth", 4)
        kwargs["xgb_learning_rate"]    = xgb.get("learning_rate", 0.05)
        kwargs["xgb_subsample"]        = xgb.get("subsample", 0.8)
        kwargs["xgb_colsample_bytree"] = xgb.get("colsample_bytree", 0.8)

    else:
        raise ValueError(
            f"Unknown classifier_type={classifier_type!r}. "
            "Expected 'logistic', 'random_forest', or 'xgboost'."
        )

    return kwargs


def _run_one_step(X_train, X_test, Y_train, Y_test,
                  n_dims, explained_variance, is_logistic, binary,
                  classifier_type="logistic", logistic_C=1.0,
                  rf_n_estimators=300, rf_max_depth=None, rf_min_samples_leaf=1,
                  xgb_n_estimators=300, xgb_max_depth=4, xgb_learning_rate=0.05,
                  xgb_subsample=0.8, xgb_colsample_bytree=0.8):
    """
    Standardize, fit, evaluate.
    Return (clf_or_reg, scaler, results_train, results_test).

    classifier_type : "logistic" | "random_forest" | "xgboost"
        Ignored when is_logistic=False (linear regression is always used then).
    """
    scaler = reg.fit_scaler(X_train)
    Xtr = scaler.transform(X_train).astype(np.float64)
    Xte = scaler.transform(X_test).astype(np.float64)

    if is_logistic:
        multi_class = not binary

        if classifier_type == "logistic":
            clf = reg.fit_logistic(Xtr, Y_train, multi_class=multi_class, C=logistic_C)
        elif classifier_type == "random_forest":
            clf = reg.fit_random_forest(
                Xtr, Y_train,
                n_estimators=rf_n_estimators,
                max_depth=rf_max_depth,
                min_samples_leaf=rf_min_samples_leaf,
            )
        elif classifier_type == "xgboost":
            clf = reg.fit_xgboost(
                Xtr, Y_train,
                n_estimators=xgb_n_estimators,
                max_depth=xgb_max_depth,
                learning_rate=xgb_learning_rate,
                subsample=xgb_subsample,
                colsample_bytree=xgb_colsample_bytree,
            )
        else:
            raise ValueError(
                f"Unknown classifier_type={classifier_type!r}. "
                "Expected 'logistic', 'random_forest', or 'xgboost'."
            )

        eval_fn = reg.eval_classifier_binary if binary else reg.eval_classifier_multiclass
        r_train = eval_fn(clf, Xtr, Y_train, n_dims, explained_variance)
        r_test  = eval_fn(clf, Xte, Y_test,  n_dims, explained_variance)
        return clf, scaler, r_train, r_test

    else:
        # Linear regression — classifier_type is ignored
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

def _run_pca_source(cfg, Y_full, client, args):
    n_train       = cfg["n_train"]
    n_test        = N_PATIENTS - n_train
    special_split = cfg.get("special_split")
    pca_run_id    = cfg["pca_run_id"]
    y_name        = cfg["y_name"]
    binary        = cfg.get("group_binary", False) and y_name == "group"
    is_logistic   = y_name == "group"

    # ── Read params from PCA run once (avoid redundant API calls) ────────────
    pca_params       = client.get_run(pca_run_id).data.params
    frame_tag        = pca_params.get("frame_tag", "ED")
    use_both_frames  = frame_tag == "ED+ES"
    frame_type       = "ED" if use_both_frames else frame_tag
    stratify_ongroup = pca_params.get("stratify_ongroup", "False") == "True"

    # ── Load X ───────────────────────────────────────────────────────────────
    print("Loading image data...")
    X_train, _, X_test, split_name = loader.load_numpy_splits(
        source_folder=pca_params["source_folder"],
        cache_folder=pca_params.get("cache_folder", "X_vectors"),
        n_train=n_train, n_val=0, n_test=n_test,
        special_split=special_split,
        stratify_ongroup=stratify_ongroup,
        use_both_frames=use_both_frames,
        frame_type=frame_type,
        image_roi_only=pca_params.get("image_roi_only", "True") == "True",
        recalculate=False,
        recalculate_y=cfg.get("recalculate_y", True), 
        y_cache_folder=cfg.get("y_cache_folder", "Y_vectors"), 
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
    Y_train, _, Y_test = _apply_split_to_Y(Y_full, n_train, 0, n_test, special_split, stratify_ongroup,  
                                                                                      recalculate_y=cfg.get("recalculate_y", True), y_cache_folder=cfg.get("y_cache_folder", "Y_vectors"))
    if use_both_frames:
        Y_train = np.concatenate([Y_train, Y_train])
        Y_test  = np.concatenate([Y_test,  Y_test])
    if binary:
        bin_val = cfg["group_bin_value"]
        Y_train = (Y_train == bin_val).astype(int)    
        Y_test  = (Y_test  == bin_val).astype(int)

    # ── Classifier kwargs (dispatch logistic / RF / XGB) ─────────────────────
    classifier_type = cfg.get("classifier_type", "logistic")  
    clf_kwargs = _get_classifier_kwargs(cfg, use_both_frames=use_both_frames)

    # ── Sweep over latent dims ────────────────────────────────────────────────
    latent_dims_list = sorted(set(cfg["latent_dims_list"]))
    n_pc_confusion   = cfg.get("n_pc_confusion", 12)
    run_label        = cfg.get("experiment_tag", "baseline")

    results_test_all  = []
    results_train_all = []

    tags = {"trial_id": args.trial_id} if args.trial_id else None 
    with tracking.start_run("regression", _run_name(cfg, split_name), tags=tags): 
        tracking.log_params(_build_params(cfg, split_name))
        tracking.log_artifact(CONFIG_PATH)

        for n_pc in latent_dims_list:
            if n_pc > pca.n_components_:
                print(f"WARNING: n_pc={n_pc} > pca.n_components_={pca.n_components_}, skipping")
                continue

            cumvar = float(np.sum(pca.explained_variance_ratio_[:n_pc]))
            Xtr    = X_train_pca[:, :n_pc]
            Xte    = X_test_pca[:,  :n_pc]

            print(f"  → fitting n_pc={n_pc} with {classifier_type}...") 

            _, _, r_train, r_test = _run_one_step(
                Xtr, Xte, Y_train, Y_test,
                n_pc, cumvar, is_logistic, binary,
                **clf_kwargs,
            )
            _log_step_metrics(r_train, r_test, step=n_pc, is_logistic=is_logistic, binary=binary)
            _save_result_json(r_train, r_test, n_pc, label="pc")

            results_train_all.append(r_train)
            results_test_all.append(r_test)
            print(f"  n_pc={n_pc:4d} | cumvar={cumvar:.3f} | "
                  + _result_summary(r_test, is_logistic, binary))

        results_train_all.sort(key=lambda r: r["n_dims"])
        results_test_all.sort(key=lambda r: r["n_dims"])

        _save_plots(results_train_all, results_test_all,
                    n_pc_confusion, y_name, run_label, is_logistic, binary)


# ── AE source ─────────────────────────────────────────────────────────────────

def _run_ae_source(cfg, Y_full, client, args):

    n_train       = cfg["n_train"]
    n_val         = cfg.get("n_val", 0)                    
    n_test        = N_PATIENTS - n_train - n_val            
    eval_on       = cfg.get("eval_on", "test")              
    special_split = cfg.get("special_split")
    split_name    = _derive_split_name(special_split)
    y_name        = cfg["y_name"]
    binary        = cfg.get("group_binary", False) and y_name == "group"
    is_logistic   = y_name == "group"
    ae_cfg        = cfg["ae_source"] or {}

    if eval_on not in ("test", "val"):
        raise ValueError(f"eval_on must be 'test' or 'val', got {eval_on!r}")
    if eval_on == "val" and n_val == 0:
        raise ValueError("eval_on='val' requires n_val > 0 in regression.yaml")

    # ── Search AE runs ────────────────────────────────────────────────────────
    ae_filter_params = {}  
    if args.ae_trial_tag:
        # Mode Agent : find AE run with tag
        conditions = [f"tags.trial_id = '{args.ae_trial_tag}'"]
        for kv in args.ae_filter:
            if "=" not in kv:
                raise SystemExit(f"--ae-filter expects KEY=VALUE, got: {kv!r}")
            k, v = kv.split("=", 1)
            conditions.append(f"params.{k} = '{v}'")
            ae_filter_params[k] = v 
        filter_str = " AND ".join(conditions)
    else:
        # Manual (default) mode
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
    ref_params       = runs_by_latdim[sorted_latdims[0]][0]
    frame_tag        = ref_params.get("params.frame_tag", "ED")
    use_both         = frame_tag == "ED+ES"
    frame_type       = "ED" if use_both else frame_tag
    image_roi_only   = ref_params.get("params.image_roi_only", "True") == "True"
    source_folder    = ref_params.get("params.source_folder", "registered_frames")
    stratify_ongroup = ref_params.get("params.stratify_ongroup", "False") == "True"

    # Verify all AE runs share the same split
    for latdim in sorted_latdims:
            ae_run_id = runs_by_latdim[latdim][0]["run_id"]
            _verify_split(ae_run_id, {"split_name": split_name, "n_train": n_train, "n_val": n_val, 
                                    "stratify_ongroup": stratify_ongroup}, client)
    print(f"Split verified against all {len(sorted_latdims)} AE runs ✓")

    # ── Load tensor datasets (same split as AE training) ─────────────────────
    print("Loading image data (tensor format for AE encoding)...")
    train_ds, val_ds, test_ds, _, _ = loader.load_tensor_datasets(
        source_folder=source_folder,
        cache_folder="X_vectors",
        n_train=n_train, n_val=n_val, n_test=n_test,   
        special_split=special_split,
        stratify_ongroup=stratify_ongroup,
        use_both_frames=use_both,
        frame_type=frame_type,
        image_roi_only=image_roi_only,
        recalculate=False,
        recalculate_y=cfg.get("recalculate_y", True),
        y_cache_folder=cfg.get("y_cache_folder", "Y_vectors"),
    )
    print(f"Split: {split_name} | train={len(train_ds)} | test={len(test_ds)}")

    # ── Y (duplicate if both frames) ─────────────────────────────────────────
    Y_train_base, Y_val_base, Y_test_base = _apply_split_to_Y(Y_full, n_train, n_val, n_test, special_split, stratify_ongroup,
                                                                                          recalculate_y=cfg.get("recalculate_y", True), y_cache_folder=cfg.get("y_cache_folder", "Y_vectors"))
    if use_both:
        Y_train_base = np.concatenate([Y_train_base, Y_train_base])
        Y_val_base   = np.concatenate([Y_val_base,   Y_val_base]) if n_val > 0 else Y_val_base
        Y_test_base  = np.concatenate([Y_test_base,  Y_test_base])
    if binary:
        bin_val = cfg["group_bin_value"]
        Y_train_base = (Y_train_base == bin_val).astype(int)
        if n_val > 0:
            Y_val_base = (Y_val_base == bin_val).astype(int)
        Y_test_base = (Y_test_base == bin_val).astype(int)

    eval_ds     = val_ds if eval_on == "val" else test_ds
    Y_eval_base = Y_val_base if eval_on == "val" else Y_test_base

    # ── Classifier kwargs (dispatch logistic / RF / XGB) ─────────────────────
    clf_kwargs = _get_classifier_kwargs(cfg, use_both_frames=use_both)

    device         = aet.get_device()
    n_pc_confusion = cfg.get("n_pc_confusion", sorted_latdims[len(sorted_latdims) // 2])
    run_label      = cfg.get("experiment_tag", "baseline")

    results_train_all = []
    results_test_all  = []

    tags = {"trial_id": args.trial_id} if args.trial_id else None   
    with tracking.start_run("regression", _run_name(cfg, split_name), tags=tags):   
        tracking.log_params({**_build_params(cfg, split_name, args), "eval_on": eval_on, **ae_filter_params})
        tracking.log_artifact(CONFIG_PATH)

        for latdim in sorted_latdims:
            ae_run_id  = runs_by_latdim[latdim][0]["run_id"]
            ae_row     = runs_by_latdim[latdim][0]
            model_name = ae_row["params.model_name"]
            dropout    = float(ae_row.get("params.dropout_rate", 0.0))
            best_epoch = int(
                ae_row.get("params.best_epoch") or ae_row.get("params.n_epochs", 100)
            )

            model   = _load_ae_model(ae_run_id, model_name, latdim, dropout, best_epoch, device, client)
            Z_train = _encode_dataset(model, train_ds, device)
            Z_eval  = _encode_dataset(model, eval_ds,  device)         
            print(f"  latent_dim={latdim} | encoded train={Z_train.shape} eval({eval_on})={Z_eval.shape}")

            _, _, r_train, r_test = _run_one_step(
                Z_train, Z_eval, Y_train_base, Y_eval_base,
                latdim, None, is_logistic, binary,
                **clf_kwargs,
            )
            _log_step_metrics(r_train, r_test, step=latdim, is_logistic=is_logistic, binary=binary)
            _save_result_json(r_train, r_test, latdim, label="latdim")

            results_train_all.append(r_train)
            results_test_all.append(r_test)
            print(f"  latent_dim={latdim:3d} | " + _result_summary(r_test, is_logistic, binary))

        _save_plots(results_train_all, results_test_all,
                    n_pc_confusion, y_name, run_label, is_logistic, binary)


# ── Result JSON helpers ───────────────────────────────────────────────────────

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
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"results_{n_dims}{label}.json"
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
    src             = cfg["source_type"]
    tag             = cfg.get("experiment_tag", "baseline")
    y               = cfg["y_name"]
    classifier_type = cfg.get("classifier_type", "logistic")
    return f"regression_{src}_{cfg['n_train']}patients_{split_name}_{y}_{classifier_type}_{tag}"


def _build_params(cfg, split_name, args):
    classifier_type = cfg.get("classifier_type", "logistic")
    params = {
        "source_type":     cfg["source_type"],
        "y_name":          cfg["y_name"],
        "group_binary":    str(cfg.get("group_binary", False)),
        "group_bin_value": cfg.get("group_bin_value", ""),
        "n_train":         cfg["n_train"],
        "n_test":          N_PATIENTS - cfg["n_train"],
        "split_name":      split_name,
        "experiment_tag":  cfg.get("experiment_tag", "baseline"),
        "n_pc_confusion":  cfg.get("n_pc_confusion", 12),
        "classifier_type": classifier_type,
    }

    # Hyperparamètres spécifiques au classifieur
    if classifier_type == "logistic":
        params["logistic_C"] = cfg.get("logistic_C", 1.0)
    elif classifier_type == "random_forest":
        rf = cfg.get("random_forest") or {}
        params["rf_n_estimators"]     = rf.get("n_estimators", 300)
        params["rf_max_depth"]        = str(rf.get("max_depth", None))
        params["rf_min_samples_leaf"] = rf.get("min_samples_leaf", 1)
    elif classifier_type == "xgboost":
        xgb = cfg.get("xgboost") or {}
        params["xgb_n_estimators"]     = xgb.get("n_estimators", 300)
        params["xgb_max_depth"]        = xgb.get("max_depth", 4)
        params["xgb_learning_rate"]    = xgb.get("learning_rate", 0.05)
        params["xgb_subsample"]        = xgb.get("subsample", 0.8)
        params["xgb_colsample_bytree"] = xgb.get("colsample_bytree", 0.8)

    if cfg["source_type"] == "pca":
        params["pca_run_id"] = cfg["pca_run_id"]
    elif args and args.ae_trial_tag:      # Agent mode         
        params["ae_trial_tag"] = args.ae_trial_tag
        params["ae_filter"]    = ";".join(args.ae_filter) if args.ae_filter else ""
    else:                                           
        ae = cfg["ae_source"]
        params["ae_model_name"]     = ae["model_name"]
        params["ae_experiment_tag"] = ae["experiment_tag"]
        params["ae_split_name"]     = ae.get("split_name", split_name)

    return params


# ── Plot-only mode ────────────────────────────────────────────────────────────

def _plot_only(cfg, client):
    """Reload result JSON artifacts from a previous run and regenerate plots."""
    load_run_id    = cfg["load_run_id"]
    run_params     = client.get_run(load_run_id).data.params

    source_type    = run_params["source_type"]
    y_name         = run_params["y_name"]
    binary         = run_params["group_binary"] == "True"
    is_logistic    = y_name == "group"
    run_label      = run_params.get("experiment_tag", "baseline")
    n_pc_confusion = cfg.get("n_pc_confusion") or int(run_params.get("n_pc_confusion", 12))

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


def _nearest_n_pc(available: list[int], target: int) -> int:
    """Return the value in `available` closest to `target` on a log scale."""
    return min(available, key=lambda n: abs(np.log(n) - np.log(target)))


def _check_consistent_runs(runs_params: list[dict]):
    """Warn (non-blocking) if loaded regression runs differ on params other than split."""
    keys = ["n_train", "source_type"]
    if runs_params[0].get("source_type") == "ae":
        keys += ["ae_model_name", "ae_experiment_tag"]

    for k in keys:
        values = {p.get(k) for p in runs_params}
        if len(values) > 1:
            print(f"WARNING: param '{k}' differs across selected runs: {values}")


def _plot_compare(cfg, client):
    """Average confusion matrices from several regression runs (different splits)."""
    run_ids     = cfg["load_run_ids"]
    n_pc_target = cfg["n_pc_confusion"]

    runs_params  = []
    cms          = []
    classes      = None
    used_run_ids = []
    label        = None

    for run_id in run_ids:
        run_params  = client.get_run(run_id).data.params
        source_type = run_params.get("source_type")
        run_label   = "pc" if source_type == "pca" else "latdim"

        available = _list_result_dims(run_id, run_label, client)
        if not available:
            print(f"WARNING: run {run_id} has no saved results — skipped")
            continue

        chosen = n_pc_target if n_pc_target in available else _nearest_n_pc(available, n_pc_target)
        if chosen != n_pc_target:
            print(f"WARNING: run {run_id} has no n_pc={n_pc_target}, "
                  f"using closest available: n_pc={chosen}")

        _, r_test = _load_result_json(run_id, chosen, run_label, client)
        if "confusion_matrix" not in r_test:
            print(f"WARNING: run {run_id} (n_pc={chosen}) has no confusion matrix — skipped "
                  f"(y_name must be 'group', non-binary)")
            continue

        if classes is None:
            classes = r_test["classes"]
            label   = run_label
        elif classes != r_test["classes"]:
            raise ValueError(f"Run {run_id} has classes {r_test['classes']} != {classes}")

        cms.append(r_test["confusion_matrix"])
        runs_params.append(run_params)
        used_run_ids.append(run_id)

    if not cms:
        raise ValueError(f"No usable confusion matrices found for n_pc≈{n_pc_target} among {run_ids}")

    _check_consistent_runs(runs_params)

    y_name        = runs_params[0]["y_name"]
    source_type   = runs_params[0]["source_type"]
    n_splits      = len(cms)
    run_label_tag = cfg.get("experiment_tag", "compare")

    print(f"compare_mode | {n_splits}/{len(run_ids)} runs used | "
          f"source={source_type} | n_pc≈{n_pc_target} | y={y_name} | runs={used_run_ids}")

    title   = (f"Averaged confusion matrix — {y_name} — {source_type} "
               f"— n_pc≈{n_pc_target} — {n_splits} splits")
    outpath = (RESULTS_FOLDER /
               f"confusionmatrix_average_{source_type}_{n_splits}splits_"
               f"{n_pc_target}{label}_{run_label_tag}.png")

    rgp.plot_average_confusion_matrix(cms, classes, title, outpath)
    print(f"Saved: {outpath}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = _parse_args()
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    if args.ae_trial_tag and cfg["source_type"] != "ae":  
        raise SystemExit(
            f"--ae-trial-tag only applies to source_type='ae' (got {cfg['source_type']!r}).  Regression cancelled because regression.yaml inconsistent with tag"
        )

    tracking._setup()
    client = mlflow.MlflowClient()

    y_name        = cfg["y_name"]
    n_train       = cfg["n_train"]
    special_split = cfg.get("special_split")
    split_name    = _derive_split_name(special_split)

    if cfg.get("compare_mode"):
        if not cfg.get("load_run_ids"):
            raise ValueError("compare_mode requires load_run_ids in regression.yaml")
        _plot_compare(cfg, client)
        return

    if cfg.get("plot_only"):
        if not cfg.get("load_run_id"):
            raise ValueError("plot_only requires load_run_id in regression.yaml")
        _plot_only(cfg, client)
        return

    classifier_type = cfg.get("classifier_type", "logistic")
    print(f"Regression | source={cfg['source_type']} | y={y_name} | "
          f"classifier={classifier_type} | n_train={n_train} | split={split_name}")

    # ── Load Y (all patients, split applied inside each source function) ──────
    Y_full = load_patient_metadata(
                                                                        y_name, N_PATIENTS,
                                                                        recalculate=cfg.get("recalculate_y", True),
                                                                        cache_folder=cfg.get("y_cache_folder", "Y_vectors"),
                                                                        )
    print(f"Y loaded: {y_name}, {len(Y_full)} patients")

    if cfg["source_type"] == "pca":
        if not cfg.get("pca_run_id"):
            raise ValueError("pca_run_id must be set in regression.yaml for source_type='pca'")
        _run_pca_source(cfg, Y_full, client, args)

    elif cfg["source_type"] == "ae":
        if not args.ae_trial_tag and not cfg.get("ae_source"):  
            raise ValueError(
                "ae_source must be set in regression.yaml for source_type='ae' "
                "unless --ae-trial-tag is provided (agent mode)."
            )
        _run_ae_source(cfg, Y_full, client, args)

    else:
        raise ValueError(f"Unknown source_type: {cfg['source_type']!r}. Use 'pca' or 'ae'.")


if __name__ == "__main__":
    main()