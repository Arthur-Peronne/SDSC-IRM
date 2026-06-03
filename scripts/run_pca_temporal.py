# scripts/run_pca_temporal.py
"""
Temporal PCA on a single patient's 4D cardiac MRI.

For each patient, all 3D volumes (timeframes) are samples — the PCA captures
how voxel intensities co-vary over time (cardiac cycle dynamics).

Reads configuration from configs/pca_temporal.yaml.
Results (model + plots) are tracked in MLflow under experiment "pca_temporal".

Usage:
    python scripts/run_pca_temporal.py
"""

import yaml
import joblib
import nibabel as nib
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA

from src.config import RESULTS_FOLDER
from src.data import importdata as ipd
from src.models import pca as pc
from src.visualization import mri_plots as mrp
from src import tracking

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "pca_temporal.yaml"


def main():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    patient_id  = cfg["patient_id"]
    patient_str = f"patient{patient_id:03d}"
    run_name    = f"PCA_temporal_{patient_str}_{cfg['experiment_tag']}"

    with tracking.start_run("pca_temporal", run_name):

        # ── Config artifact + params ──────────────────────────────────────────
        tracking.log_artifact(CONFIG_PATH)
        tracking.log_params({
            "patient_id":           patient_id,
            "max_pc_calc":          cfg["max_pc_calc"],
            "recalculate_pca":      cfg["recalculate_pca"],
            "experiment_tag":       cfg["experiment_tag"],
            "n_pc_to_reconstruct":  cfg["n_pc_to_reconstruct"],
            "eigenvectors_to_plot": str(cfg["eigenvectors_to_plot"]),
        })

        # ── Load 4D nii ───────────────────────────────────────────────────────
        nii_path   = ipd.get_patient_acdc_path(patient_id, file_type="4d")
        nii_obj    = nib.load(nii_path)
        data_array = nii_obj.get_fdata()
        print(f"Loaded {patient_str} 4D: shape {data_array.shape}")

        # X: (n_epochs, n_voxels) — sklearn PCA centers column-wise (per voxel)
        X = pc.pca1_transpose(data_array, print_infos=False)

        # ── PCA ───────────────────────────────────────────────────────────────
        if cfg["recalculate_pca"]:
            pca = PCA(n_components=min(nii_obj.shape[3], cfg["max_pc_calc"]))
            X_reduced = pca.fit_transform(X)
            tracking.log_sklearn_model(pca, filename=f"{patient_str}_pca.joblib")
            print(f"PCA calculated: {pca.n_components_} components")
        else:
            run_id = cfg.get("load_run_id")
            if not run_id:
                raise ValueError("recalculate_pca: false requires load_run_id in YAML")
            import mlflow
            local_path = mlflow.MlflowClient().download_artifacts(run_id, f"{patient_str}_pca.joblib")
            pca = joblib.load(local_path)
            X_reduced = pca.transform(X)
            print(f"PCA loaded from run {run_id}")

        # ── Metrics ───────────────────────────────────────────────────────────
        for i, var in enumerate(pca.explained_variance_ratio_):
            tracking.log_metric(f"explained_variance_pc{i + 1}", float(var))
        tracking.log_metric(
            f"cumulative_variance_pc{pca.n_components_}",
            float(np.sum(pca.explained_variance_ratio_)),
        )

        # ── Plot : explained variance ─────────────────────────────────────────
        if cfg["do_explained_variance"]:
            pc.plot_pca_explipower(pca, patient_str)
            tracking.log_artifact(RESULTS_FOLDER / f"{patient_str}_PCA_explainedvariance.png")

        # ── Plot : PC values in eigenbase ─────────────────────────────────────
        if cfg["do_pc_in_eigenbase"]:
            pc_max = cfg["pc_max"]
            if pc_max % 2 != 0:
                raise ValueError(f"pc_max must be even, got {pc_max}")
            for i in range(0, min(pc_max, pca.n_components_ - 1), 2):
                pc.plot_pcvalues_2d(X_reduced, i, i + 1, patient_str, "_pc_in_eigenbase", axisscale_fixed=False)
                tracking.log_artifact(RESULTS_FOLDER / f"{patient_str}_pc_in_eigenbase_{i + 1}and{i + 2}.png")

        # ── Plot : eigenvectors ───────────────────────────────────────────────
        if cfg["do_eigenvec_plot"]:
            shape_3d = data_array.shape[:3]
            for n in cfg["eigenvectors_to_plot"]:
                idx = n - 1  # YAML uses 1-indexed
                if idx >= pca.n_components_:
                    print(f"[SKIP] eigenvector {n} > n_components ({pca.n_components_})")
                    continue
                nii_eig = pc.eigenvector_to_nii(pca.components_[idx], shape_3d, nii_obj)
                mrp.plot_oneimg(nii_eig, patient_str=patient_str, file_str="eigenvector", details_str=f"pc{n}")
                tracking.log_artifact(RESULTS_FOLDER / f"{patient_str}_eigenvector_pc{n}.png")

        # ── Plot : reconstruction ─────────────────────────────────────────────
        if cfg["do_reconstruction"]:
            n_pc   = cfg["n_pc_to_reconstruct"]
            frames = cfg["frames_to_reconstruct"]
            nii_rec = pc.pca1_reconstruct(X_reduced, pca, n_pc, data_array, nii_obj)

            if frames == "all":
                mrp.plot_allepochs(nii_rec, patient_str, suffix=f"_reconstructed_{n_pc}pc")
                mrp.plot_allepochs(nii_obj, patient_str, suffix="_original")
            else:
                for t in frames:
                    nii_frame_rec  = mrp.get_oneepoch(nii_rec,  t)
                    nii_frame_orig = mrp.get_oneepoch(nii_obj,  t)
                    mrp.plot_oneimg(nii_frame_rec,  patient_str=patient_str, file_str=f"frame{t:02d}", details_str=f"reconstructed_{n_pc}pc")
                    mrp.plot_oneimg(nii_frame_orig, patient_str=patient_str, file_str=f"frame{t:02d}", details_str="original")
                    tracking.log_artifact(RESULTS_FOLDER / f"{patient_str}_frame{t:02d}_reconstructed_{n_pc}pc.png")
                    tracking.log_artifact(RESULTS_FOLDER / f"{patient_str}_frame{t:02d}_original.png")


if __name__ == "__main__":
    main()
