# src/models/regression.py
"""
Pure regression functions for PCA- and AE-based cardiac classification
and metadata prediction. No file I/O — all results returned as dicts.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
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
from xgboost import XGBClassifier


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


def fit_random_forest(X_train: np.ndarray, Y_train: np.ndarray,
                      n_estimators: int = 300,
                      max_depth: int | None = None,
                      min_samples_leaf: int = 1) -> RandomForestClassifier:
    """
    Train a Random Forest classifier (multiclass or binary).

    Parameters
    ----------
    X_train : np.ndarray, shape (n, n_features)
    Y_train : np.ndarray, shape (n,)
    n_estimators : int
        Number of trees. 300 is a safe default; more rarely hurts but slows training.
    max_depth : int or None
        Maximum tree depth. None = trees grow until leaves are pure (may overfit
        on small datasets; try 5–15 if overfitting is observed).
    min_samples_leaf : int
        Minimum samples per leaf. Increasing this (e.g. 2–5) regularises the forest
        on small datasets like ACDC (n=100 train).

    Notes
    -----
    StandardScaler is NOT required for Random Forest (invariant to feature scaling),
    but the scaler is still applied upstream in _run_one_step for consistency.
    Feature importances are available on the returned object via .feature_importances_.
    """
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_jobs=-1,          # use all available CPU cores
    )
    clf.fit(X_train, Y_train)
    return clf


class _XGBClassifierWithStringLabels:
    """
    Thin wrapper around XGBClassifier that handles string labels transparently.

    XGBoost requires integer targets; this wrapper encodes labels to integers
    before fitting and exposes .classes_, .predict(), .predict_proba() in the
    same way as RandomForestClassifier and LogisticRegression, so that
    eval_classifier_* works without any changes in calling code.
    """

    def __init__(self, **xgb_kwargs):
        from sklearn.preprocessing import LabelEncoder
        self._clf = XGBClassifier(**xgb_kwargs)
        self._le  = LabelEncoder()

    def fit(self, X, Y):
        Y_encoded = self._le.fit_transform(Y)
        self._clf.fit(X, Y_encoded)
        return self

    @property
    def classes_(self):
        return self._le.classes_       # e.g. array(['DCM', 'HCM', 'MINF', 'NOR', 'RV'])

    def predict(self, X):
        return self._le.inverse_transform(self._clf.predict(X))

    def predict_proba(self, X):
        return self._clf.predict_proba(X)


def fit_xgboost(X_train: np.ndarray, Y_train: np.ndarray,
                n_estimators: int = 300,
                max_depth: int = 4,
                learning_rate: float = 0.05,
                subsample: float = 0.8,
                colsample_bytree: float = 0.8) -> _XGBClassifierWithStringLabels:
    """
    Train an XGBoost classifier (multiclass or binary).

    Parameters
    ----------
    X_train : np.ndarray, shape (n, n_features)
    Y_train : np.ndarray, shape (n,)
    n_estimators : int
        Number of boosting rounds. More rounds = more expressive but risks overfitting.
    max_depth : int
        Max tree depth per round. Keep low (3–6) on small datasets to avoid overfitting.
    learning_rate : float
        Step size shrinkage. Lower = slower but more robust; pair with high n_estimators.
    subsample : float
        Fraction of training samples used per boosting round (row subsampling).
        Values < 1 add stochasticity and reduce overfitting.
    colsample_bytree : float
        Fraction of features used per boosting round (column subsampling).
        Similar to the random feature selection in Random Forests.

    Returns
    -------
    _XGBClassifierWithStringLabels
        Wrapper exposing .classes_, .predict(), .predict_proba() with string labels,
        compatible with eval_classifier_binary and eval_classifier_multiclass.
    """
    clf = _XGBClassifierWithStringLabels(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=42,
        n_jobs=-1,
        eval_metric="mlogloss",
        verbosity=0,
    )
    clf.fit(X_train, Y_train)
    return clf


def fit_linear(X_train: np.ndarray, Y_train: np.ndarray) -> LinearRegression:
    """Train a linear regression model."""
    reg = LinearRegression()
    reg.fit(X_train, Y_train)
    return reg


def eval_classifier_binary(clf, X_test: np.ndarray, Y_test: np.ndarray,
                            n_dims: int, explained_variance: float | None) -> dict:
    """
    Evaluate any binary sklearn-compatible classifier and return metrics as a dict.

    Works for LogisticRegression, RandomForestClassifier, XGBClassifier, etc.
    All three expose .predict() and .predict_proba() with the same signatures.
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


def eval_classifier_multiclass(clf, X_test: np.ndarray, Y_test: np.ndarray,
                                n_dims: int, explained_variance: float | None) -> dict:
    """
    Evaluate any multiclass sklearn-compatible classifier and return metrics as a dict.

    Works for LogisticRegression, RandomForestClassifier, XGBClassifier, etc.
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


# ── Backward-compatible aliases (kept so existing plot_only runs still work) ──
eval_logistic_binary     = eval_classifier_binary
eval_logistic_multiclass = eval_classifier_multiclass


def eval_linear(reg, X_test: np.ndarray, Y_test: np.ndarray,
                n_dims: int, explained_variance: float | None) -> dict:
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