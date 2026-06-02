# scripts/run_registration_pipelinecheck.py
"""
Visual quality check of the preprocessing pipeline.

For a selection of patients, plots images and masks at each step:
  1. Resampled (full image)
  2. Cropped (around cardiac centroid)
  3. Registered (aligned to reference)

Also computes Dice scores before/after registration, separately for ED and ES frames.

Run this script after any change to resampling, cropping, or registration
to verify the pipeline produces correct outputs.
"""

import csv
import glob
import nibabel as nib
import numpy as np
from pathlib import Path

from src.config import PROCESSED_IMAGES_FOLDER, RESULTS_FOLDER
from src.visualization import mri_plots as mrp
from src.data import registration as rgt

# ── User choices : VISUAL CHECKS ─────────────────────────────────────────────
do_plot_checks = False

# ED frames to check per patient (patient090 uses frame04 as ED)
patients_ED = {
    "patient001": "frame01",
    "patient050": "frame01",
    "patient090": "frame04",
    "patient110": "frame01",
    "patient140": "frame01",
}

# ES frames to check per patient
patients_ES = {
    "patient001": "frame12",
    "patient050": "frame12",
    "patient090": "frame11",
    "patient110": "frame12",
    "patient140": "frame12",
}

check_ED       = True    # plot ED frames
check_ES       = True    # plot ES frames
check_resampled  = True
check_cropped    = True
check_registered = True

# ── User choices : DICE ───────────────────────────────────────────────────────
do_dice_checks      = True
registered_folder   = "registered_frames"
registered_OLD      = None  # set to None to skip OLD comparison
n_worst_to_print    = 5

# ── Folder paths ──────────────────────────────────────────────────────────────
resampled_folder_path  = PROCESSED_IMAGES_FOLDER / "resampled_frames"
cropped_folder_path    = PROCESSED_IMAGES_FOLDER / "cropped_frames"
registered_folder_path = PROCESSED_IMAGES_FOLDER / registered_folder

# ── Helper ────────────────────────────────────────────────────────────────────
def _load(path, label):
    path = Path(path)
    if not path.exists():
        print(f"  [MISSING] {label}: {path}")
        return None
    return nib.load(path)


def _check_patient(patient, frame):
    print(f"\n{'='*60}")
    print(f"Patient: {patient} | Frame: {frame}")
    print(f"{'='*60}")

    if check_resampled:
        print("  Checking resampled...")
        img  = _load(resampled_folder_path / f"{patient}_{frame}_resampled.nii.gz",    "resampled img")
        mask = _load(resampled_folder_path / f"{patient}_{frame}_resampled_gt.nii.gz", "resampled mask")
        if img is not None:
            print(f"    Shape: {img.shape} | Spacing: {img.header.get_zooms()}")
            mrp.plot_oneimg(img,  patient_str=patient, file_str=frame, details_str="resampled")
        if mask is not None:
            mrp.plot_onemask(mask, patient_str=patient, file_str=frame, details_str="resampled_gt")
        if img is not None and mask is not None:
            mrp.plot_oneimagemask(img, mask, patient_str=patient, file_str=frame, details_str="resampled")

    if check_cropped:
        print("  Checking cropped...")
        img  = _load(cropped_folder_path / f"{patient}_{frame}_cropped.nii.gz",    "cropped img")
        mask = _load(cropped_folder_path / f"{patient}_{frame}_cropped_gt.nii.gz", "cropped mask")
        if img is not None:
            print(f"    Shape: {img.shape} | Spacing: {img.header.get_zooms()}")
            mrp.plot_oneimg(img,  patient_str=patient, file_str=frame, details_str="cropped")
        if mask is not None:
            mrp.plot_onemask(mask, patient_str=patient, file_str=frame, details_str="cropped_gt")
        if img is not None and mask is not None:
            mrp.plot_oneimagemask(img, mask, patient_str=patient, file_str=frame, details_str="cropped")

    if check_registered:
        print("  Checking registered...")
        img  = _load(registered_folder_path / f"{patient}_{frame}_registered.nii.gz",    "registered img")
        mask = _load(registered_folder_path / f"{patient}_{frame}_registered_gt.nii.gz", "registered mask")
        if img is not None:
            print(f"    Shape: {img.shape} | Spacing: {img.header.get_zooms()}")
            mrp.plot_oneimg(img,  patient_str=patient, file_str=frame, details_str="registered")
        if mask is not None:
            mrp.plot_onemask(mask, patient_str=patient, file_str=frame, details_str="registered_gt")
        if img is not None and mask is not None:
            mrp.plot_oneimagemask(img, mask, patient_str=patient, file_str=frame, details_str="registered")


def _save_dice_csv(results, label):
    """Save per-patient Dice results to CSV in RESULTS_FOLDER."""
    filename = RESULTS_FOLDER / f"dice_{label.replace(' ', '_')}.csv"
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["patient_id", "frame_id", "dice_before", "dice_after", "dice_gain"])
        writer.writeheader()
        writer.writerows(results)
    print(f"  Saved: {filename}")


def _save_dice_summary(stats, label):
    """Save Dice summary statistics to TXT in RESULTS_FOLDER."""
    filename = RESULTS_FOLDER / f"dice_summary_{label.replace(' ', '_')}.txt"
    with open(filename, "w") as f:
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
    print(f"  Saved: {filename}")


def _print_dice_stats(label, results, save=True):
    stats = rgt.stats_dice(results)
    print(f"\n--- {label} ({len(results)} frames) ---")
    print(f"Dice before : mean={stats['mean_dice_before']:.3f} | median={stats['median_dice_before']:.3f}")
    print(f"Dice after  : mean={stats['mean_dice_after']:.3f}  | median={stats['median_dice_after']:.3f}")
    print(f"Dice gain   : mean={stats['mean_dice_gain']:.3f}")
    print(f"Improved: {stats['n_improved']} | Equal: {stats['n_equal']} | Worse: {stats['n_worse']}")
    if save:
        _save_dice_csv(results, label)
        _save_dice_summary(stats, label)
    return stats


# ── Visual checks ─────────────────────────────────────────────────────────────
if do_plot_checks:
    if check_ED:
        for patient, frame in patients_ED.items():
            _check_patient(patient, frame)

    if check_ES:
        for patient, frame in patients_ES.items():
            _check_patient(patient, frame)

    print("\nDone. Check results folder for output images.")

# ── Dice checks ───────────────────────────────────────────────────────────────
if do_dice_checks:
    print("\nComputing Dice scores before/after registration...")
    results_all = rgt.dice_all_patients(registered_folder=registered_folder)

    _print_dice_stats("All frames", results_all)

    # Worst patients
    results_sorted = sorted(results_all, key=lambda r: r["dice_after"])
    print(f"\nWorst {n_worst_to_print} patients after registration:")
    for r in results_sorted[:n_worst_to_print]:
        print(f"  {r['patient_id']} {r['frame_id']} : dice_after={r['dice_after']:.3f} | gain={r['dice_gain']:+.3f}")

    # ED only
    results_ED = [r for r in results_all if r["frame_id"] in ("frame01", "frame04")]
    _print_dice_stats("ED frames only", results_ED)

    # ES only
    results_ES = [r for r in results_all if r["frame_id"] not in ("frame01", "frame04")]
    _print_dice_stats("ES frames only", results_ES)

    # OLD registration comparison
    if registered_OLD is not None:
        new_reg_names = {
            Path(p).name.replace("_registered_gt.nii.gz", "")
            for p in glob.glob(str(PROCESSED_IMAGES_FOLDER / registered_folder / "patient*_frame*_registered_gt.nii.gz"))
        }
        old_reg_paths = sorted(glob.glob(
            str(PROCESSED_IMAGES_FOLDER / registered_OLD / "patient*_frame*_registered_gt.nii.gz")
        ))
        old_reg_names = {Path(p).name.replace("_registered_gt.nii.gz", "") for p in old_reg_paths}

        # Coverage check: same frames in both folders?
        only_in_new = new_reg_names - old_reg_names
        only_in_old = old_reg_names - new_reg_names
        if only_in_new or only_in_old:
            print(f"\nWARNING: coverage mismatch between '{registered_folder}' and '{registered_OLD}'")
            if only_in_new:
                print(f"  Only in new ({len(only_in_new)}): {sorted(only_in_new)[:5]}")
            if only_in_old:
                print(f"  Only in old ({len(only_in_old)}): {sorted(only_in_old)[:5]}")
        else:
            print(f"\nCoverage OK: both folders have {len(new_reg_names)} frames.")

        # Build lookup: stem -> cropped mask path (all frames)
        all_cropped = {
            Path(p).name.replace("_cropped_gt.nii.gz", ""): p
            for p in glob.glob(str(PROCESSED_IMAGES_FOLDER / "cropped_frames/patient*_frame*_cropped_gt.nii.gz"))
        }
        ref_mask = nib.load(all_cropped["patient001_frame01"]).get_fdata()

        results_old = []
        for reg_path in old_reg_paths:
            key = Path(reg_path).name.replace("_registered_gt.nii.gz", "")
            if key not in all_cropped:
                print(f"  [MISSING cropped] {key} — skipping")
                continue
            patient_id, frame_id = key.split("_")[:2]
            crop_mask = nib.load(all_cropped[key]).get_fdata()
            reg_mask  = nib.load(reg_path).get_fdata()
            d_before  = float(rgt.dice_score(ref_mask, crop_mask))
            d_after   = float(rgt.dice_score(ref_mask, reg_mask))
            results_old.append({
                "patient_id":  patient_id,
                "frame_id":    frame_id,
                "dice_before": d_before,
                "dice_after":  d_after,
                "dice_gain":   d_after - d_before,
            })
        if results_old:
            _print_dice_stats(f"OLD registration ({len(results_old)} frames)", results_old)
        else:
            print("  No paired frames found for OLD comparison.")