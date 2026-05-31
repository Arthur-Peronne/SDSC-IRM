# src/data/loader.py
"""
Unified data loading and train/val/test split utilities.
Shared by the PCA spatial and autoencoder pipelines.

Two public entry points
-----------------------
load_numpy_splits()    → flat numpy arrays  (PCA pipeline)
load_tensor_datasets() → TensorDatasets     (AE pipeline)

Both rely on the same internal split logic via _split_frames().
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset

from src.models import pca_spatial as pcs


# ── Internal helpers ──────────────────────────────────────────────────────────

def _load_frame(source_folder, cache_folder, frame_type, image_roi_only, mask,
                binary_mask, flatten, recalculate, n_jobs):
    return pcs.get_vectorsarray(
        source_folder=source_folder,
        pca_folder=cache_folder,
        recalculate=recalculate,
        mask=mask,
        binary_mask=binary_mask,
        image_roi_only=image_roi_only,
        flatten=flatten,
        n_jobs=n_jobs,
        frame_type=frame_type,
    )


def _split_frames(X_ED, X_ES, n_development, n_validation):
    """
    Split arrays into train / val / test by patient index.

    If X_ES is provided (use_both_frames=True), ED and ES are concatenated
    within each split — both frames of a patient are always in the same split.

    Parameters
    ----------
    X_ED : np.ndarray, shape (n_total_patients, ...)
    X_ES : np.ndarray or None
    n_development : int   — train + validation patients
    n_validation : int    — 0 = no validation set

    Returns
    -------
    X_train, X_val, X_test : np.ndarray
        X_val is empty (shape (0, ...)) when n_validation = 0.
    """
    n_train = n_development - n_validation

    def _combine(start, end):
        sub_ed = X_ED[start:end]
        if X_ES is None:
            return sub_ed
        return np.concatenate([sub_ed, X_ES[start:end]], axis=0)

    return (
        _combine(0, n_train),
        _combine(n_train, n_development),
        _combine(n_development, None),
    )


# ── Public API ────────────────────────────────────────────────────────────────

def load_numpy_splits(
    source_folder: str,
    cache_folder: str,
    n_development: int,
    n_validation: int,
    use_both_frames: bool = True,
    frame_type: str = "ED",
    image_roi_only: bool = True,
    mask: bool = False,
    binary_mask: bool = False,
    recalculate: bool = False,
    n_jobs: int = -1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load flat numpy arrays and return train / val / test splits.
    For use in the PCA spatial pipeline.

    Parameters
    ----------
    source_folder : str
        Subfolder in PROCESSED_IMAGES_FOLDER containing registered NIfTI files.
    cache_folder : str
        Subfolder where .npy arrays are cached.
    n_development : int
        Number of patients used for train + validation.
    n_validation : int
        Number of validation patients (0 = no validation set).
    use_both_frames : bool
        True  → load ED and ES, concatenate within each split.
        False → load only frame_type.
    frame_type : str
        "ED" or "ES" — used only if use_both_frames=False.
    image_roi_only : bool
        Zero out voxels outside the ROI mask.
    mask : bool
        Load GT mask files instead of images.
    binary_mask : bool
        Binarize loaded data.
    recalculate : bool
        Recompute from NIfTI files even if a .npy cache exists.
    n_jobs : int
        Parallel workers for NIfTI loading (-1 = all cores).

    Returns
    -------
    X_train, X_val, X_test : np.ndarray, shape (n, n_voxels)
        X_val has shape (0, n_voxels) when n_validation = 0.
    """
    kwargs = dict(
        source_folder=source_folder, cache_folder=cache_folder,
        image_roi_only=image_roi_only, mask=mask, binary_mask=binary_mask,
        flatten=True, recalculate=recalculate, n_jobs=n_jobs,
    )

    X_ED = _load_frame(frame_type="ED", **kwargs)
    X_ES = None

    if use_both_frames:
        X_ES = _load_frame(frame_type="ES", **kwargs)
        if X_ED.shape != X_ES.shape:
            raise ValueError(f"Shape mismatch: X_ED={X_ED.shape} vs X_ES={X_ES.shape}")
    elif frame_type != "ED":
        X_ED = _load_frame(frame_type=frame_type, **kwargs)

    return _split_frames(X_ED, X_ES, n_development, n_validation)


def load_tensor_datasets(
    source_folder: str,
    cache_folder: str,
    n_development: int,
    n_validation: int,
    use_both_frames: bool = True,
    image_roi_only: bool = True,
    percentile_max: float = 99.9,
    recalculate: bool = False,
    n_jobs: int = 1,
) -> tuple:
    """
    Load 3D arrays, normalize, split, and wrap into TensorDatasets.
    For use in the autoencoder pipeline.

    Normalisation is computed on the ED development pool only (stable reference),
    then applied to all splits and both frames.

    Parameters
    ----------
    source_folder : str
    cache_folder : str
    n_development : int
    n_validation : int
    use_both_frames : bool
    image_roi_only : bool
    percentile_max : float
        Percentile used for max-normalization (default 99.9).
    recalculate : bool
    n_jobs : int

    Returns
    -------
    train_dataset : TensorDataset
    val_dataset : TensorDataset or None
        None when n_validation = 0.
    test_dataset : TensorDataset
    X_maxnorm : float
        Normalization constant — needed to denormalize reconstructions.
    """
    kwargs = dict(
        source_folder=source_folder, cache_folder=cache_folder,
        image_roi_only=image_roi_only, mask=False, binary_mask=False,
        flatten=False, recalculate=recalculate, n_jobs=n_jobs,
    )

    X_ED = _load_frame(frame_type="ED", **kwargs)
    X_ES = None

    if use_both_frames:
        X_ES = _load_frame(frame_type="ES", **kwargs)
        if X_ED.shape != X_ES.shape:
            raise ValueError(f"Shape mismatch: X_ED={X_ED.shape} vs X_ES={X_ES.shape}")

    X_maxnorm = float(np.percentile(X_ED[:n_development], percentile_max))

    def _normalize(X):
        return np.clip(X, 0, X_maxnorm) / X_maxnorm

    X_ED = _normalize(X_ED)
    if X_ES is not None:
        X_ES = _normalize(X_ES)

    X_train, X_val, X_test = _split_frames(X_ED, X_ES, n_development, n_validation)

    def _to_tensor_dataset(X):
        X = np.transpose(X, (0, 3, 1, 2))       # (N, D, H, W) = (N, 32, 128, 128)
        X = X[:, np.newaxis, :, :, :]            # (N, 1, 32, 128, 128)
        X = X.astype(np.float32, copy=False)
        return TensorDataset(torch.from_numpy(X))

    train_dataset = _to_tensor_dataset(X_train)
    val_dataset = _to_tensor_dataset(X_val) if n_validation > 0 else None
    test_dataset = _to_tensor_dataset(X_test)

    return train_dataset, val_dataset, test_dataset, X_maxnorm
