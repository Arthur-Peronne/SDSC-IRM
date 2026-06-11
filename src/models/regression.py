# src/models/regression.py
"""
Pure regression functions for PCA- and AE-based cardiac classification
and metadata prediction. No file I/O — all results returned as dicts.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    precision_score,
    recall_score,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)


def n_pc_for_variance(pca, threshold: float) -> int:
    """
    Return the number of PCs needed to reach a given cumulative explained
    variance threshold.  threshold >= 1.0 (e.g. 2.0) returns all components.
    """
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    idx = np.where(cumulative >= threshold)[0]
    return int(len(cumulative)) if len(idx) == 0 else int(idx[0] + 1)


def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    """Fit a StandardScaler on X_train and return it."""
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def fit_logistic(X_train: np.ndarray, Y_train: np.ndarray,
                 multi_class: bool = True, C: float = 1.0) -> LogisticRegression:
    """
    Train a logistic regression classifier (binary or multiclass).

    Parameters
    ----------
    X_train : np.ndarray, shape (n, n_features), float64
    Y_train : np.ndarray, shape (n,)
    multi_class : bool
        True  → multinomial (5 ACDC groups).
        False → binary (one class vs rest).
    C : float
        Inverse of regularization strength. Use 0.5 when training on ED+ES
        (correlated pairs) to compensate for the inflated apparent sample size.
    """
    max_iter = 30000 if multi_class else 10000
    clf = LogisticRegression(C=C, max_iter=max_iter, random_state=42, solver="newton-cg")
    clf.fit(X_train, Y_train)
    return clf


def fit_linear(X_train: np.ndarray, Y_train: np.ndarray) -> LinearRegression:
    """Train a linear regression model."""
    reg = LinearRegression()
    reg.fit(X_train, Y_train)
    return reg


def eval_logistic_binary(clf, X_test: np.ndarray, Y_test: np.ndarray,
                         n_dims: int, explained_variance: float) -> dict:
    """
    Evaluate a binary logistic classifier and return metrics as a dict.

    Parameters
    ----------
    clf : fitted LogisticRegression
    X_test : np.ndarray
    Y_test : np.ndarray
    n_dims : int
        Number of latent dimensions used (for logging).
    explained_variance : float
        Cumulative explained variance kept (PCA); 1.0 for AE.
    """
    Y_pred = clf.predict(X_test)
    Y_prob = clf.predict_proba(X_test)[:, 1]

    return {
        "n_dims": n_dims,
        "explained_variance": explained_variance,
        "accuracy": float(accuracy_score(Y_test, Y_pred)),
        "roc_auc": float(roc_auc_score(Y_test, Y_prob)),
        "precision": float(precision_score(Y_test, Y_pred, zero_division=0)),
        "recall": float(recall_score(Y_test, Y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(Y_test, Y_pred),
        "classes": [int(c) for c in clf.classes_],
    }


def eval_logistic_multiclass(clf, X_test: np.ndarray, Y_test: np.ndarray,
                              n_dims: int, explained_variance: float) -> dict:
    """
    Evaluate a multiclass logistic classifier and return metrics as a dict.
    """
    Y_pred = clf.predict(X_test)

    precision_per_class = precision_score(
        Y_test, Y_pred, average=None, labels=clf.classes_, zero_division=0
    )
    recall_per_class = recall_score(
        Y_test, Y_pred, average=None, labels=clf.classes_, zero_division=0
    )

    return {
        "n_dims": n_dims,
        "explained_variance": explained_variance,
        "accuracy": float(accuracy_score(Y_test, Y_pred)),
        "precision_macro": float(precision_score(Y_test, Y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(Y_test, Y_pred, average="macro", zero_division=0)),
        "precision_per_class": {str(c): float(v) for c, v in zip(clf.classes_, precision_per_class)},
        "recall_per_class": {str(c): float(v) for c, v in zip(clf.classes_, recall_per_class)},
        "confusion_matrix": confusion_matrix(Y_test, Y_pred, labels=clf.classes_),
        "classes": [str(c) for c in clf.classes_],
    }


def eval_linear(reg, X_test: np.ndarray, Y_test: np.ndarray,
                n_dims: int, explained_variance: float) -> dict:
    """
    Evaluate a linear regression model and return metrics as a dict.
    """
    Y_pred = reg.predict(X_test)

    return {
        "n_dims": n_dims,
        "explained_variance": explained_variance,
        "r2": float(r2_score(Y_test, Y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(Y_test, Y_pred))),
        "mae": float(mean_absolute_error(Y_test, Y_pred)),
        "Y_pred": Y_pred,
        "Y_test": Y_test,
    }
