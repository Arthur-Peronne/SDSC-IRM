# src/data/loader.py
"""
Unified data loading and train/val/test split utilities.
Shared by the PCA spatial and autoencoder pipelines.

Two public entry points
-----------------------
load_numpy_splits()    → flat numpy arrays  (PCA pipeline)
load_tensor_datasets() → TensorDatasets     (AE pipeline)

Both rely on the same internal split logic via _split_frames().
Split indices are computed by src.data.splits.get_split_indices().
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset

from src.data import splits as splt
from src.models import pca_spatial as pcs
from src.data.importdata import load_patient_metadata

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


def _split_frames(
    X_ED: np.ndarray,
    X_ES: np.ndarray | None,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split arrays into train / val / test using patient indices.

    If X_ES is provided (use_both_frames=True), ED and ES are concatenated
    within each split — both frames of a patient are always in the same split.

    Parameters
    ----------
    X_ED : np.ndarray, shape (n_patients, ...)
    X_ES : np.ndarray or None
    train_idx, val_idx, test_idx : np.ndarray of int
        Patient indices returned by splits.get_split_indices().
        val_idx / test_idx may be empty arrays.

    Returns
    -------
    X_train, X_val, X_test : np.ndarray
        X_val / X_test have shape (0, ...) when the corresponding index array
        is empty.
    """
    def _combine(idx):
        sub_ed = X_ED[idx]
        if X_ES is None:
            return sub_ed
        return np.concatenate([sub_ed, X_ES[idx]], axis=0)

    return _combine(train_idx), _combine(val_idx), _combine(test_idx)


# ── Public API ────────────────────────────────────────────────────────────────

def load_numpy_splits(
    source_folder: str,
    cache_folder: str,
    n_train: int,
    n_val: int,
    n_test: int,
    special_split: str | None = None,
    stratify_ongroup: bool = False,
    use_both_frames: bool = True,
    frame_type: str = "ED",
    image_roi_only: bool = True,
    mask: bool = False,
    binary_mask: bool = False,
    recalculate: bool = False,
    n_jobs: int = -1,
    recalculate_y: bool = True,            
    y_cache_folder: str = "Y_vectors",      
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Load flat numpy arrays and return train / val / test splits.
    For use in the PCA spatial pipeline.

    Parameters
    ----------
    source_folder : str
        Subfolder in PROCESSED_IMAGES_FOLDER containing registered NIfTI files.
    cache_folder : str
        Subfolder where .npy arrays are cached.
    n_train : int
        Number of training patients.
    n_val : int
        Number of validation patients (0 = no validation set).
    n_test : int
        Number of test patients.  n_train + n_val + n_test must equal 150.
    special_split : str or None
        None       → sequential split, logged as "splitdefault".
        "split42"  → seeded random split, logged under that name.
    stratify_ongroup : bool
        If True (and special_split is not None), stratify the split on the
        ACDC group of each patient. Ignored for the sequential split.
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
        X_val / X_test have shape (0, n_voxels) when n_val / n_test = 0.
    split_name : str
        Effective split name to log in MLflow ("splitdefault" or special_split).
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

    strat_array = (
        load_patient_metadata(
            "group", len(X_ED),
            recalculate=recalculate_y, cache_folder=y_cache_folder, 
        ) if stratify_ongroup else None
    )

    train_idx, val_idx, test_idx, split_name = splt.get_split_indices(
        n_train=n_train, n_val=n_val, n_test=n_test,
        special_split=special_split,
        stratify=strat_array,
        n_patients=len(X_ED),
    )
    X_train, X_val, X_test = _split_frames(X_ED, X_ES, train_idx, val_idx, test_idx)
    return X_train, X_val, X_test, split_name


def load_tensor_datasets(
    source_folder: str,
    cache_folder: str,
    n_train: int,
    n_val: int,
    n_test: int,
    special_split: str | None = None,
    stratify_ongroup: bool = False,
    use_both_frames: bool = True,
    frame_type: str = "ED",
    image_roi_only: bool = True,
    mask: bool = False,          
    binary_mask: bool = False,  
    percentile_max: float = 99.9,
    recalculate: bool = False,
    n_jobs: int = 1,
    recalculate_y: bool = True, 
    y_cache_folder: str = "Y_vectors", 
) -> tuple[TensorDataset, TensorDataset | None, TensorDataset, float, str]:
    """
    Load 3D arrays, normalize, split, and wrap into TensorDatasets.
    For use in the autoencoder pipeline.

    Normalisation is computed on the ED training pool only (stable reference),
    then applied to all splits and both frames.

    Parameters
    ----------
    source_folder : str
    cache_folder : str
    n_train : int
        Number of training patients.
    n_val : int
        Number of validation patients (0 = no validation set).
    n_test : int
        Number of test patients.  n_train + n_val + n_test must equal 150.
    special_split : str or None
        None       → sequential split, logged as "splitdefault".
        "split42"  → seeded random split, logged under that name.
    stratify_ongroup : bool
        If True (and special_split is not None), stratify the split on the
        ACDC group of each patient. Ignored for the sequential split.
    use_both_frames : bool
    frame_type : str
        "ED" or "ES" — used only if use_both_frames=False.
    image_roi_only : bool
    percentile_max : float
        Percentile used for max-normalization (default 99.9).
    recalculate : bool
    n_jobs : int

    Returns
    -------
    train_dataset : TensorDataset
    val_dataset : TensorDataset or None
        None when n_val = 0.
    test_dataset : TensorDataset
    X_maxnorm : float
        Normalization constant — needed to denormalize reconstructions.
    split_name : str
        Effective split name to log in MLflow ("splitdefault" or special_split).
    """
    kwargs = dict(
        source_folder=source_folder, cache_folder=cache_folder,
        image_roi_only=image_roi_only, mask=mask, binary_mask=binary_mask,
        flatten=False, recalculate=recalculate, n_jobs=n_jobs,
    )

    X_ED = _load_frame(frame_type="ED", **kwargs)
    X_ES = None

    if use_both_frames:
        X_ES = _load_frame(frame_type="ES", **kwargs)
        if X_ED.shape != X_ES.shape:
            raise ValueError(f"Shape mismatch: X_ED={X_ED.shape} vs X_ES={X_ES.shape}")
    elif frame_type != "ED":
        X_ED = _load_frame(frame_type=frame_type, **kwargs)

    print(f"special_split = {special_split}")
    strat_array = (
        load_patient_metadata("group", len(X_ED, 
                                                            recalculate=recalculate_y, cache_folder=y_cache_folder)) 
        if (stratify_ongroup and special_split is not None) else None
    )

    train_idx, val_idx, test_idx, split_name = splt.get_split_indices(
        n_train=n_train, n_val=n_val, n_test=n_test,
        special_split=special_split,
        stratify=strat_array,
        n_patients=len(X_ED),
    )

    # Normalisation reference: always computed on the training pool
    # (for ES-only mode, X_ED actually contains ES frames at this point)
    X_maxnorm = float(np.percentile(X_ED[train_idx], percentile_max))

    def _normalize(X):
        return np.clip(X, 0, X_maxnorm) / X_maxnorm

    X_ED = _normalize(X_ED)
    if X_ES is not None:
        X_ES = _normalize(X_ES)

    X_train, X_val, X_test = _split_frames(X_ED, X_ES, train_idx, val_idx, test_idx)

    def _to_tensor_dataset(X):
        X = np.transpose(X, (0, 3, 1, 2))       # (N, D, H, W) = (N, 32, 128, 128)
        X = X[:, np.newaxis, :, :, :]            # (N, 1, 32, 128, 128)
        X = X.astype(np.float32, copy=False)
        return TensorDataset(torch.from_numpy(X))

    train_dataset = _to_tensor_dataset(X_train)
    val_dataset   = _to_tensor_dataset(X_val) if n_val > 0 else None
    test_dataset  = _to_tensor_dataset(X_test)

    return train_dataset, val_dataset, test_dataset, X_maxnorm, split_name
