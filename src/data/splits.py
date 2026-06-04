# src/data/splits.py
"""
Patient-level train / val / test split utilities.
Shared by the PCA spatial, autoencoder, and regression pipelines.

Two modes
---------
- Default sequential split  (special_split=None) : patients are assigned in
  order — first n_train for training, then n_val, then n_test.  Logged as
  "splitdefault" in MLflow.
- Named random split        (special_split="split42") : the name is hashed to
  a deterministic seed, patients are shuffled, and the same partition is
  always reproduced from the same name.

Usage
-----
    from src.data.splits import get_split_indices

    train_idx, val_idx, test_idx, split_name = get_split_indices(
        n_train=100, n_val=20, n_test=30,
        special_split=None,      # → "splitdefault"
    )

    train_idx, val_idx, test_idx, split_name = get_split_indices(
        n_train=100, n_val=20, n_test=30,
        special_split="split42",
    )
"""

import hashlib

import numpy as np

DEFAULT_SPLIT_NAME = "splitdefault"


def splitname_to_seed(splitname: str, modulo: int = 2**32) -> int:
    """Deterministic seed from a split name (SHA-256, first 8 hex chars)."""
    digest = hashlib.sha256(splitname.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % modulo


def get_split_indices(
    n_train: int,
    n_val: int,
    n_test: int,
    special_split: str | None = None,
    stratify: np.ndarray | None = None,
    n_patients: int = 150,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Return patient-level split indices and the effective split name.

    Parameters
    ----------
    n_train : int
        Number of training patients.
    n_val : int
        Number of validation patients (0 = no validation set).
    n_test : int
        Number of test patients (0 = no test set).
    special_split : str or None
        None       → sequential split, logged as "splitdefault".
        "split42"  → seeded random split, logged under that name.
    stratify : np.ndarray or None
        Optional array of class labels (length n_patients) for stratified
        sampling.  Only used when special_split is not None.
        Ignored for the sequential split.
    n_patients : int
        Total number of patients.  Default 150.  Must equal
        n_train + n_val + n_test exactly.

    Returns
    -------
    train_idx, val_idx, test_idx : np.ndarray of int
        Patient indices into the full array.
        val_idx / test_idx are empty arrays (shape (0,)) when n_val / n_test = 0.
    split_name : str
        Effective split name to log in MLflow.

    Raises
    ------
    ValueError
        If n_train + n_val + n_test != n_patients.
    """
    if n_train + n_val + n_test != n_patients:
        raise ValueError(
            f"n_train ({n_train}) + n_val ({n_val}) + n_test ({n_test}) "
            f"= {n_train + n_val + n_test} ≠ n_patients ({n_patients})"
        )

    all_idx = np.arange(n_patients)

    # ── Default sequential split ──────────────────────────────────────────────
    if special_split is None:
        train_idx = all_idx[:n_train]
        val_idx   = all_idx[n_train : n_train + n_val]
        test_idx  = all_idx[n_train + n_val :]
        return train_idx, val_idx, test_idx, DEFAULT_SPLIT_NAME

    # ── Named random split ────────────────────────────────────────────────────
    if stratify is not None:
        from sklearn.model_selection import train_test_split

        remaining = all_idx
        strat_rem = np.asarray(stratify)

        if n_test > 0:
            remaining, test_idx, strat_rem, _ = train_test_split(
                remaining, strat_rem,
                test_size=n_test,
                random_state=splitname_to_seed(special_split),
                stratify=strat_rem,
            )
        else:
            test_idx = np.array([], dtype=int)

        if n_val > 0:
            train_idx, val_idx = train_test_split(
                remaining,
                test_size=n_val,
                random_state=splitname_to_seed(special_split),
                stratify=strat_rem,
            )
        else:
            train_idx = remaining
            val_idx   = np.array([], dtype=int)

        return (
            np.sort(train_idx),
            np.sort(val_idx),
            np.sort(test_idx),
            special_split,
        )

    seed     = splitname_to_seed(special_split)
    rng      = np.random.default_rng(seed)
    shuffled = rng.permutation(all_idx)
    train_idx = np.sort(shuffled[:n_train])
    val_idx   = np.sort(shuffled[n_train : n_train + n_val])
    test_idx  = np.sort(shuffled[n_train + n_val :])
    return train_idx, val_idx, test_idx, special_split
