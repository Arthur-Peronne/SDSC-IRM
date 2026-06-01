# scripts/run_visualize.py
"""
Visualize MRI NIfTI images (raw ACDC or processed).
Reads configuration from configs/visualization.yaml.
Outputs PNG files to RESULTS_FOLDER.

Supported sources:
  raw       - original ACDC frames or 4D volumes from DATADIR
  processed - resampled/cropped/registered files from PROCESSED_IMAGES_FOLDER

Frame numbers (ED/ES) are read from each patient's Info.cfg.
"""

import yaml
import nibabel as nib
from pathlib import Path

from src.config import DATADIR, PROCESSED_IMAGES_FOLDER
from src.data.importdata import read_info_cfg
from src.visualization import mri_plots as mrp

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "visualization.yaml"

# Maps processed subfolder name to file suffix
_FOLDER_TO_SUFFIX = {
    "registered_frames": "_registered",
    "resampled_frames": "_resampled",
    "cropped_frames": "_cropped",
}


def _get_subset(patient_id: int) -> str:
    return "training" if patient_id <= 100 else "testing"


def _read_frame_numbers(patient_id: int) -> dict:
    """Return ED and ES frame numbers from Info.cfg."""
    patient_str = f"patient{patient_id:03d}"
    cfg_path = DATADIR / _get_subset(patient_id) / patient_str / "Info.cfg"
    info = read_info_cfg(cfg_path)
    return {"ED": int(info["ED"]), "ES": int(info["ES"])}


def _build_raw_paths(patient_id: int, raw_file_type: str, frame_num: int) -> tuple[Path, Path | None]:
    """Return (image_path, mask_path_or_None) for a raw ACDC file."""
    patient_str = f"patient{patient_id:03d}"
    base = DATADIR / _get_subset(patient_id) / patient_str
    if raw_file_type == "4d":
        return base / f"{patient_str}_4d.nii.gz", None
    img = base / f"{patient_str}_frame{frame_num:02d}.nii.gz"
    mask = base / f"{patient_str}_frame{frame_num:02d}_gt.nii.gz"
    return img, mask


def _build_processed_paths(patient_id: int, folder: str, suffix: str, frame_num: int) -> tuple[Path, Path]:
    """Return (image_path, mask_path) for a processed file."""
    patient_str = f"patient{patient_id:03d}"
    base = PROCESSED_IMAGES_FOLDER / folder
    img = base / f"{patient_str}_frame{frame_num:02d}{suffix}.nii.gz"
    mask = base / f"{patient_str}_frame{frame_num:02d}{suffix}_gt.nii.gz"
    return img, mask


def _plot_frame(patient_id: int, frame_type: str, frame_num: int, cfg: dict) -> None:
    source = cfg["source"]
    plot_modes = set(cfg.get("plot_modes", ["image"]))
    patient_str = f"patient{patient_id:03d}"

    if source == "raw":
        path_img, path_mask = _build_raw_paths(patient_id, cfg["raw_file_type"], frame_num)
    else:
        folder = cfg["folder"]
        suffix = _FOLDER_TO_SUFFIX.get(folder, f"_{folder.replace('_frames', '')}")
        path_img, path_mask = _build_processed_paths(patient_id, folder, suffix, frame_num)

    if not path_img.exists():
        print(f"  [SKIP] {path_img.name} not found")
        return

    frame_str = f"frame{frame_num:02d}"
    if source == "raw":
        file_str = frame_str
        details = f"raw_{frame_type}"
    else:
        suffix_clean = suffix.lstrip("_")  # "registered", "resampled", "cropped"
        file_str = frame_str
        details = f"processed_{suffix_clean}_{frame_type}"
    
    nii_img = nib.load(path_img)

    if "image" in plot_modes:
        mrp.plot_oneimg(nii_img, patient_str=patient_str, file_str=file_str, details_str=details)
        print(f"  [OK] image ({frame_type})")

    if ("mask" in plot_modes or "overlay" in plot_modes) and path_mask is not None:
        if path_mask.exists():
            nii_mask = nib.load(path_mask)
            if "mask" in plot_modes:
                mrp.plot_onemask(nii_mask, patient_str=patient_str, file_str=file_str, details_str=f"{details}_mask")
                print(f"  [OK] mask ({frame_type})")
            if "overlay" in plot_modes:
                mrp.plot_oneimagemask(nii_img, nii_mask, patient_str=patient_str, file_str=file_str, details_str=details)
                print(f"  [OK] overlay ({frame_type})")
        else:
            print(f"  [SKIP] mask not found for {frame_type}, skipping mask/overlay")

    if "all_epochs" in plot_modes and len(nii_img.shape) > 3:
        mrp.plot_allepochs(nii_img, patient_str, epoch_limit=cfg.get("epoch_limit", 1000), suffix=f"_{frame_type}")
        print(f"  [OK] all_epochs")


def main():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    patient_id = cfg["patient_id"]
    source = cfg["source"]
    raw_file_type = cfg.get("raw_file_type", "frame")
    frame_type = cfg.get("frame_type", "ED")

    print(f"\nVisualizing patient{patient_id:03d} | source={source}", end="")
    if source == "raw":
        print(f" | file_type={raw_file_type}")
    else:
        print(f" | folder={cfg['folder']} | frame_type={frame_type}")

    # 4D raw: no frame number needed and only all_epochs plot_mode possible
    if source == "raw" and raw_file_type == "4d":
        cfg["plot_modes"] = ["all_epochs"]
        _plot_frame(patient_id, "4d", 0, cfg)
        return

    # All other cases: read frame numbers from Info.cfg
    frame_numbers = _read_frame_numbers(patient_id)
    frame_types_to_run = ["ED", "ES"] if frame_type == "both" else [frame_type]
    for ft in frame_types_to_run:
        _plot_frame(patient_id, ft, frame_numbers[ft], cfg)


if __name__ == "__main__":
    main()
