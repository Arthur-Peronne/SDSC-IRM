# scripts/run_pca_spatial.py
"""
Spatial PCA across patients.

For each patient, all voxels of one registered 3D frame are a sample — the PCA
captures how voxel intensities co-vary across the population (anatomical modes).

Reads configuration from configs/pca_spatial.yaml.
Results (model + plots) are tracked in MLflow under experiment "pca_spatial".

Usage:
    python scripts/run_pca_spatial.py
"""

import yaml
import joblib
import nibabel as nib
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA

from src.config import RESULTS_FOLDER
from src.data import importdata as ipd
from src.data import loader
from src.models import pca as pc
from src.models import pca_spatial as pcs
from src.models.pca import pca_spatial_reconstruct
from src.training import ae_training as aet
from src.visualization import ae_plots as aep
from src.visualization import pca_plots as pcp
from src import tracking

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "pca_spatial.yaml"


def main():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    # ── Derived parameters ────────────────────────────────────────────────────
    original_shape   = tuple(cfg["original_shape"])
    use_both_frames  = cfg["use_both_frames"]
    frame_type       = cfg["frame_type"]
    frame_tag        = "ED+ES" if use_both_frames else frame_type
    n_development    = cfg["n_development"]
    n_validation     = cfg["n_validation"]
    n_train          = n_development - n_validation
    n_train_images   = n_train * 2 if use_both_frames else n_train
    n_val_images     = n_validation * 2 if use_both_frames else n_validation
    splitname        = cfg["splitname"]
    recalculate      = cfg["recalculate_pca"]
    load_run_id      = cfg.get("load_run_id") if not recalculate else None

    if not recalculate and not load_run_id:
        raise ValueError("recalculate_pca: false requires load_run_id in YAML")

    pca_filename = f"pca_{frame_tag}.joblib"
    run_name     = f"PCA_{n_train_images}patients_{splitname}_{frame_tag}_{cfg['experiment_tag']}"
    run_ctx      = tracking.start_run("pca_spatial", run_name) if recalculate else tracking.resume_run(load_run_id)

    with run_ctx:

        # ── Config artifact + params (CALC only) ──────────────────────────────
        if recalculate:
            tracking.log_artifact(CONFIG_PATH)
            tracking.log_params({
                "source_folder":   cfg["source_folder"],
                "n_development":   n_development,
                "n_validation":    n_validation,
                "n_train":         n_train,
                "frame_tag":       frame_tag,
                "splitname":       splitname,
                "image_roi_only":  cfg["image_roi_only"],
                "mask_ys":         cfg["mask_ys"],
                "mask_bin":        cfg["mask_bin"],
                "original_shape":  str(original_shape),
                "max_pc_calc":     cfg["max_pc_calc"],
                "experiment_tag":  cfg["experiment_tag"],
            })

        # ── Load data ─────────────────────────────────────────────────────────
        X_train, X_val, X_test = loader.load_numpy_splits(
            source_folder=cfg["source_folder"],
            cache_folder=cfg["cache_folder"],
            n_development=n_development,
            n_validation=n_validation,
            use_both_frames=use_both_frames,
            frame_type=frame_type,
            image_roi_only=cfg["image_roi_only"],
            mask=cfg["mask_ys"],
            binary_mask=cfg["mask_bin"],
            recalculate=cfg["recalculate_x"],
        )
        print(f"X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")

        # ── Row centering ─────────────────────────────────────────────────────
        # Subtract per-patient mean to remove brightness offset artefacts.
        # sklearn PCA will additionally subtract the per-voxel mean (mean image).
        if not cfg["mask_bin"]:
            row_means_train = X_train.mean(axis=1, keepdims=True)
            row_means_val   = X_val.mean(axis=1,   keepdims=True) if len(X_val) > 0 else np.zeros((0, 1))
            row_means_test  = X_test.mean(axis=1,  keepdims=True)
            X_train -= row_means_train
            if len(X_val) > 0:
                X_val -= row_means_val
            X_test  -= row_means_test
        else:
            row_means_train = np.zeros((X_train.shape[0], 1))
            row_means_val   = np.zeros((X_val.shape[0],   1))
            row_means_test  = np.zeros((X_test.shape[0],  1))

        # ── PCA ───────────────────────────────────────────────────────────────
        if recalculate:
            pca = PCA(n_components=min(X_train.shape[0], cfg["max_pc_calc"]))
            X_train_pca = pca.fit_transform(X_train)
            tracking.log_sklearn_model(pca, filename=pca_filename)
            print(f"PCA calculated: {pca.n_components_} components")
            for i, var in enumerate(pca.explained_variance_ratio_):
                tracking.log_metric(f"explained_variance_pc{i + 1}", float(var))
            tracking.log_metric(
                f"cumulative_variance_pc{pca.n_components_}",
                float(np.sum(pca.explained_variance_ratio_)),
            )
        else:
            import mlflow
            local_path = mlflow.MlflowClient().download_artifacts(load_run_id, pca_filename)
            pca = joblib.load(local_path)
            X_train_pca = pca.transform(X_train)
            print(f"PCA loaded from run {load_run_id}")

        # Project val and test into PCA space
        X_val_pca  = pca.transform(X_val)  if len(X_val)  > 0 else np.empty((0, pca.n_components_))
        X_test_pca = pca.transform(X_test)

        # ── Build plot tag ────────────────────────────────────────────────────
        plot_tag = f"allpatients_{frame_tag}"
        if cfg["mask_ys"]:         plot_tag += "_gt"
        if cfg["mask_bin"]:        plot_tag += "_bin"
        if cfg["image_roi_only"]:  plot_tag += "_imgROIonly"

        # ── Reconstruction metrics ────────────────────────────────────────────
        if cfg["compute_metrics"]:
            latdim_list = cfg["latdim_list_pca"] or list(range(1, n_train_images + 1))
            datasets = [("train", X_train, X_train_pca, 0)]
            if n_validation > 0:
                datasets.append(("validation", X_val, X_val_pca, n_train_images))
            datasets.append(("test", X_test, X_test_pca, n_train_images + n_val_images))

            for metrics_dataset, X_flat, X_pca_sub, offset in datasets:
                for latent_dimensions in latdim_list:
                    _, summary = pcs.pca_compute_metrics(
                        X_flat=X_flat,
                        X_pca=X_pca_sub,
                        pca=pca,
                        latent_dimensions=latent_dimensions,
                        offset=offset,
                        pca_name=run_name,
                        metrics_dataset=metrics_dataset,
                        original_shape=original_shape,
                    )
                    for metric, stats in summary.items():
                        for stat, value in stats.items():
                            tracking.log_metric(
                                f"{metrics_dataset}_{metric}_{stat}",
                                value,
                                step=latent_dimensions,
                            )

            if cfg.get("plot_metrics", True):
                save_path = RESULTS_FOLDER / f"{plot_tag}_metrics_vs_latentdim.pdf"
                pcp.plot_pca_metrics_vs_latentdim(
                    run_id=tracking.active_run_id(),
                    datasets=[d for d, *_ in datasets],
                    title=run_name,
                    save_path=save_path,
                )
                tracking.log_artifact(save_path)

        # ── Plot : explained variance ─────────────────────────────────────────
        if cfg["plot_explained_variance"]:
            pcp.plot_pca_explipower(pca, plot_tag)
            tracking.log_artifact(RESULTS_FOLDER / f"{plot_tag}_PCA_explainedvariance.png")

        # ── Plot : PC values in eigenbase ─────────────────────────────────────
        if cfg["plot_pc_values"]:
            pc_max = cfg["pc_max"]
            if pc_max % 2 != 0:
                raise ValueError(f"pc_max must be even, got {pc_max}")
            for i in range(0, min(pc_max, pca.n_components_ - 1), 2):
                pcp.plot_pcvalues_2d(
                    X_train_pca, i, i + 1,
                    plot_tag, "_pc_in_eigenbase",
                    scale_str="Patient number",
                    segments=False,
                    axisscale_fixed=False,
                )
                tracking.log_artifact(RESULTS_FOLDER / f"{plot_tag}_pc_in_eigenbase_{i + 1}and{i + 2}.png")

        # ── Plot : PC values colored by metadata ──────────────────────────────
        if cfg["plot_metadata"]:
            for n1, n2 in [(cfg["pc_n1"], cfg["pc_n2"]), (2, 3), (4, 5), (6, 7), (8, 9)]:
                pcs.plot_pca_patientmeta(X_train_pca, n1, n2)

        # ── Plot : eigenvectors ───────────────────────────────────────────────
        if cfg["plot_eigenvectors"]:
            all_img, _ = ipd.load_allframes_registered(folder=cfg["source_folder"], frame_type="ED")
            nii_ref = nib.load(all_img[0])
            pcs.plot_eigenvectors(
                X_train, pca, original_shape, plot_tag, nii_ref,
                eigenvectors_to_plot=cfg["eigenvectors_to_plot"],
            )
            for n in cfg["eigenvectors_to_plot"]:
                tracking.log_artifact(RESULTS_FOLDER / f"{plot_tag}_eigenvector_pc{n}.png")

        # ── Reconstruction plots ──────────────────────────────────────────────
        if cfg["plot_reconstruction"]:
            latent_dim_plot = cfg["n_pc_to_reconstruct"]
            offset_test     = n_train_images + n_val_images

            # Compute metrics on test set at latent_dim_plot for patient selection
            all_metrics_plot = []
            for i, x_patient_flat in enumerate(X_test):
                x_recon_flat = pca_spatial_reconstruct(X_test_pca[i], pca, latent_dim_plot)
                metrics = aet.reconstruction_metrics(
                    x_true=x_patient_flat.reshape(original_shape),
                    x_pred=x_recon_flat.reshape(original_shape),
                    patient_number=offset_test + 1 + i,
                    simulation_name=run_name,
                    n_epochs=None,
                    metrics_dataset="test",
                    savemetrics=False,
                )
                all_metrics_plot.append(metrics)

            if cfg["recons_auto"]:
                selected = aep.ae_select_representative_patients(
                    all_metrics_plot,
                    use_both_frames=use_both_frames,
                    n_train_images=n_train_images,
                    n_val_images=n_val_images,
                    n_development=n_development,
                )
                patients_torecons = [(v["real_patient"], v["frame_type"]) for v in selected.values()]
            else:
                patients_torecons = [tuple(p) for p in cfg["patients_torecons_manual"]]

            pcp.pca_plotcompare_selected(
                patients_torecons=patients_torecons,
                X_train_pca=X_train_pca,
                X_val_pca=X_val_pca,
                X_test_pca=X_test_pca,
                pca=pca,
                latent_dimensions=latent_dim_plot,
                original_shape=original_shape,
                use_both_frames=use_both_frames,
                n_development=n_development,
                n_train_images=n_train_images,
                n_val_images=n_val_images,
                split_name=splitname,
                row_means_train=row_means_train,
                row_means_val=row_means_val,
                row_means_test=row_means_test,
            )


if __name__ == "__main__":
    main()
