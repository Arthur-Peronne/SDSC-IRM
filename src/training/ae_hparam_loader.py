# src/training/ae_hparam_loader.py
"""
Hyperparameter resolution for autoencoder training.

Implements 4 cases depending on hyper_automatic_values and ae_HPforarchis.yaml content:
  1. hyper_automatic_values = False
     → use values directly from autoencoder.yaml (cfg)

  2. hyper_automatic_values = True, model not in hp table
     → use 'default' block from ae_HPforarchis.yaml

  3. hyper_automatic_values = True, model in hp table with exactly 1 dim entry
     → use that single entry regardless of the requested latent_dim

  4. hyper_automatic_values = True, model in hp table with 2+ dim entries
     → interpolate (log-log for lr/weight_decay/beta, log-linear for others)
        no extrapolation: clamp to nearest boundary dim

Public API
----------
  load_hp_table(path)                           → dict (call once before the loop)
  resolve_hyperparams(model_name, latent_dim, hp_table, cfg)
      → tuple: (lr, weight_decay, dropout_rate, noise_std, patience,
                beta, beta_warmup_epochs)
        Always 7 values in this fixed order. Hyperparameters absent from the
        source fall back to cfg (or 0) so the tuple is always complete.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import yaml


# ── Hyperparameters that live on a log scale (Optuna: log=True) ──────────────
_HP_LOG_SCALE: frozenset[str] = frozenset({"lr", "weight_decay", "beta"})

# ── All recognised HP keys (used for completeness checks) ────────────────────
_ALL_HP_KEYS: tuple[str, ...] = (
    "lr", "weight_decay", "dropout_rate", "noise_std", "patience",
    "beta", "beta_warmup_epochs",
)

# ── Fixed return order of resolve_hyperparams (must match unpacking order) ────
_RETURN_ORDER: tuple[str, ...] = (
    "lr", "weight_decay", "dropout_rate", "noise_std", "patience",
    "beta", "beta_warmup_epochs",
)

# ── Path to ae_HPforarchis.yaml (relative to project root) ───────────────────
_DEFAULT_HP_TABLE_PATH = Path(__file__).parent.parent.parent / "configs" / "ae_HPforarchis.yaml"


# ─────────────────────────────────────────────────────────────────────────────
# Public functions
# ─────────────────────────────────────────────────────────────────────────────

def load_hp_table(path: Path | str | None = None) -> dict:
    """
    Read ae_HPforarchis.yaml and return its content as a plain dict.

    Call this ONCE before the training loop, then pass the result to
    resolve_hyperparams() at every (model_name, latent_dim) iteration.

    Parameters
    ----------
    path : Path or str, optional
        Path to ae_HPforarchis.yaml. Defaults to configs/ae_HPforarchis.yaml
        relative to the project root.

    Returns
    -------
    dict
        Raw YAML content, e.g.:
        {
          'AE3dAsymResSeparableV2': {'dim_8': {...}, 'dim_60': {...}, ...},
          'AE3dFCDeep': {'dim_120': {...}},
          'default': {...},
        }
    """
    p = Path(path) if path is not None else _DEFAULT_HP_TABLE_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"ae_HPforarchis.yaml not found at {p}. "
            "Create it or set hyper_automatic_values: false in autoencoder.yaml."
        )
    with open(p) as f:
        table = yaml.safe_load(f)
    if table is None:
        table = {}
    return table


def resolve_hyperparams(
    model_name: str,
    latent_dim: int,
    hp_table: dict,
    cfg: dict,
) -> tuple:
    """
    Return hyperparameter values for (model_name, latent_dim) as a fixed tuple.

    Implements the 4-case logic described in the module docstring.

    Returns
    -------
    tuple
        (lr, weight_decay, dropout_rate, noise_std, patience,
         beta, beta_warmup_epochs)
        Always 7 values in this order. Missing hyperparameters fall back to
        cfg[key] if present, else 0.

    Example
    -------
        lr, weight_decay, dropout_rate, noise_std, patience, beta, beta_warmup_epochs = \\
            resolve_hyperparams(model_name, latent_dim, hp_table, cfg)
    """
    hp = _resolve_dict(model_name, latent_dim, hp_table, cfg)
    return _dict_to_tuple(hp, cfg)


def _resolve_dict(
    model_name: str,
    latent_dim: int,
    hp_table: dict,
    cfg: dict,
) -> dict:
    """Internal: resolve hyperparameters as a dict (4-case logic)."""
    auto = cfg.get("hyper_automatic_values", False)

    # ── Case 1: manual mode ───────────────────────────────────────────────────
    if not auto:
        return _hp_from_cfg(cfg)

    # ── Cases 2-4: automatic mode ─────────────────────────────────────────────
    model_entry = hp_table.get(model_name)

    # Case 2: model not in table → use default
    if model_entry is None:
        default = hp_table.get("default")
        if default is None:
            raise KeyError(
                f"Model '{model_name}' not found in ae_HPforarchis.yaml "
                "and no 'default' block exists."
            )
        print(
            f"[hp_loader] '{model_name}' not in hp table → using default hyperparameters"
        )
        return _clean_hp(default)

    # Parse dim entries: keys are 'dim_8', 'dim_60', etc.
    dim_map = _parse_dim_map(model_entry)

    if len(dim_map) == 0:
        raise ValueError(
            f"Model '{model_name}' is in ae_HPforarchis.yaml but has no "
            "valid 'dim_<int>' entries."
        )

    # Case 3: single dim entry → use as-is
    if len(dim_map) == 1:
        only_dim, only_hp = next(iter(dim_map.items()))
        print(
            f"[hp_loader] '{model_name}' has a single Optuna point "
            f"(dim={only_dim}) → applying to dim={latent_dim} unchanged"
        )
        return _clean_hp(only_hp)

    # Case 4: 2+ dim entries → interpolate (no extrapolation)
    return _interpolate(model_name, latent_dim, dim_map)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _hp_from_cfg(cfg: dict) -> dict:
    """Extract HP values from autoencoder.yaml (case 1)."""
    hp = {}
    for key in _ALL_HP_KEYS:
        if key in cfg:
            hp[key] = cfg[key]
    return hp


def _dict_to_tuple(hp: dict, cfg: dict) -> tuple:
    """
    Convert a resolved HP dict into the fixed-order tuple.

    Missing keys fall back to cfg[key] if present, else 0.
    Applies correct types (int for patience / beta_warmup_epochs, float otherwise).
    """
    out = []
    for key in _RETURN_ORDER:
        if key in hp:
            val = hp[key]
        elif key in cfg:
            val = cfg[key]
        else:
            val = 0
        if key in ("patience", "beta_warmup_epochs"):
            out.append(int(val))
        else:
            out.append(float(val))
    return tuple(out)


def _clean_hp(raw: dict) -> dict:
    """Return only recognised HP keys from a raw dict, with correct types."""
    hp = {}
    for key in _ALL_HP_KEYS:
        if key in raw:
            val = raw[key]
            # patience and beta_warmup_epochs are ints
            if key in ("patience", "beta_warmup_epochs"):
                hp[key] = int(val)
            else:
                hp[key] = float(val)
    return hp


def _parse_dim_map(model_entry: dict) -> dict[int, dict]:
    """
    Convert {'dim_8': {...}, 'dim_60': {...}} → {8: {...}, 60: {...}}.
    Non-dim_ keys are silently ignored.
    """
    dim_map: dict[int, dict] = {}
    for key, val in model_entry.items():
        if key.startswith("dim_"):
            try:
                d = int(key[4:])
                dim_map[d] = val
            except ValueError:
                warnings.warn(f"Skipping malformed key '{key}' in ae_HPforarchis.yaml")
    return dict(sorted(dim_map.items()))  # sorted by dim for clarity


def _interpolate(model_name: str, latent_dim: int, dim_map: dict[int, dict]) -> dict:
    """
    Interpolate hyperparameters for latent_dim given 2+ Optuna reference points.

    - log-log   interpolation for HP in _HP_LOG_SCALE  (lr, weight_decay, beta)
    - log-linear interpolation for all other HP

    No extrapolation: dims outside [min_dim, max_dim] are clamped to the
    nearest boundary values.
    """
    sorted_dims = sorted(dim_map.keys())
    min_dim, max_dim = sorted_dims[0], sorted_dims[-1]

    # Clamp: no extrapolation
    if latent_dim <= min_dim:
        if latent_dim < min_dim:
            print(
                f"[hp_loader] dim={latent_dim} < min Optuna dim ({min_dim}) "
                f"for '{model_name}' → clamping to dim={min_dim}"
            )
        return _clean_hp(dim_map[min_dim])

    if latent_dim >= max_dim:
        if latent_dim > max_dim:
            print(
                f"[hp_loader] dim={latent_dim} > max Optuna dim ({max_dim}) "
                f"for '{model_name}' → clamping to dim={max_dim}"
            )
        return _clean_hp(dim_map[max_dim])

    # Exact match: no interpolation needed
    if latent_dim in dim_map:
        print(
            f"[hp_loader] '{model_name}' dim={latent_dim}: exact Optuna point"
        )
        return _clean_hp(dim_map[latent_dim])

    # Find the two bracketing dims
    lower_dim = max(d for d in sorted_dims if d <= latent_dim)
    upper_dim = min(d for d in sorted_dims if d >= latent_dim)

    # t in [0, 1] on the log(dim) axis
    log_lower = math.log(lower_dim)
    log_upper = math.log(upper_dim)
    log_target = math.log(latent_dim)
    t = (log_target - log_lower) / (log_upper - log_lower)

    lower_hp = dim_map[lower_dim]
    upper_hp = dim_map[upper_dim]

    # Collect all HP keys present in at least one boundary point
    all_keys = set(lower_hp.keys()) | set(upper_hp.keys())

    result: dict[str, Any] = {}
    for key in all_keys:
        if key not in lower_hp or key not in upper_hp:
            # Key present in only one boundary: use whichever exists
            val = lower_hp.get(key, upper_hp.get(key))
            warnings.warn(
                f"[hp_loader] HP '{key}' missing in one boundary for '{model_name}' "
                f"(dims {lower_dim}/{upper_dim}) — using available value {val}"
            )
            result[key] = val
            continue

        v_lower = lower_hp[key]
        v_upper = upper_hp[key]

        if key in _HP_LOG_SCALE:
            # log-log: interpolate log(hp) linearly in log(dim)
            interp = math.exp(math.log(v_lower) + t * (math.log(v_upper) - math.log(v_lower)))
        else:
            # log-linear: interpolate hp linearly in log(dim)
            interp = v_lower + t * (v_upper - v_lower)

        result[key] = interp

    print(
        f"[hp_loader] '{model_name}' dim={latent_dim}: interpolated between "
        f"dim={lower_dim} and dim={upper_dim} (t={t:.3f})"
    )
    return _clean_hp(result)