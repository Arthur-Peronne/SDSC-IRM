# scripts/run_registration.py
"""
Preprocessing pipeline: resampling -> cropping -> registration.
All parameters are read from configs/registration.yaml.
Outputs go to processed_images/ (no MLflow tracking for preprocessing).
"""

import yaml
from pathlib import Path

from src.data import registration as rgt
from src.data import geometry as rgg
from src.data import resampling as rsp


cfg_path = Path(__file__).parent.parent / "configs" / "registration.yaml"
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

if cfg["resample_all"]:
    rsp.resample_all(
        target_spacing=cfg["target_spacing"],
        only01=cfg["only_01"],
        limit=cfg["limit"],
    )

if cfg["crop_all"]:
    rgg.crop_all_frames(
        only01=cfg["only_01"],
        crop_shape=tuple(cfg["crop_shape"]),
        limit=cfg["limit"],
    )

if cfg["register_all"]:
    rgt.register_all_frames(
        reference_patient=cfg["reference_patient"],
        reference_frame=cfg["reference_frame"],
        crop_after_registration=cfg["crop_after_registration"],
        crop_size_after_registration=tuple(cfg["crop_shape"]),
        number_of_iterations=cfg["n_iterations"],
        limit=cfg["limit"],
    )
