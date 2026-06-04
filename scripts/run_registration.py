# scripts/run_registration.py
"""
Preprocessing pipeline: resampling -> cropping -> registration.
All parameters are read from configs/registration.yaml.
Outputs go to processed_images/ (no MLflow tracking for preprocessing).

Pipeline checks (visual + Dice) can be run at the end by setting
run_pipelinechecks: true in the YAML. To run checks only, set
resample_all, crop_all, register_all to false.
"""

import csv
import glob
import nibabel as nib
import yaml
from pathlib import Path

from src.config import PROCESSED_IMAGES_FOLDER, RESULTS_FOLDER
from src.data import registration as rgt
from src.data import geometry as rgg
from src.data import resampling as rsp
from src.visualization import mri_plots as mrp


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


# ── Pipeline checks ───────────────────────────────────────────────────────────
if cfg["run_pipelinechecks"]:
    check_resampled        = cfg["check_resampled"]
    check_cropped          = cfg["check_cropped"]
    check_registered       = cfg["check_registered"]
    registered_OLD         = cfg["registered_OLD"]   # str or None
    n_worst_to_print       = cfg["n_worst_to_print"]

    resampled_folder_path  = PROCESSED_IMAGES_FOLDER / "resampled_frames"
    cropped_folder_path    = PROCESSED_IMAGES_FOLDER / "cropped_frames"
    registered_folder_path = PROCESSED_IMAGES_FOLDER / "registered_frames"

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
        filename = RESULTS_FOLDER / f"dice_{label.replace(' ', '_')}.csv"
        with open(filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["patient_id", "frame_id", "dice_before", "dice_after", "dice_gain"])
            writer.writeheader()
            writer.writerows(results)
        print(f"  Saved: {filename}")

    def _save_dice_summary(stats, label):
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

    # ── Visual checks ─────────────────────────────────────────────────────────
    if cfg["check_ED"]:
        for patient, frame in cfg["patients_ED"].items():
            _check_patient(patient, frame)
    if cfg["check_ES"]:
        for patient, frame in cfg["patients_ES"].items():
            _check_patient(patient, frame)
    if cfg["check_ED"] or cfg["check_ES"]:
        print("\nDone. Check results folder for output images.")

    # ── Dice checks ───────────────────────────────────────────────────────────
    if cfg["do_dice_checks"]:
        print("\nComputing Dice scores before/after registration...")
        results_all = rgt.dice_all_patients(registered_folder="registered_frames")

        _print_dice_stats("All frames", results_all)

        results_sorted = sorted(results_all, key=lambda r: r["dice_after"])
        print(f"\nWorst {n_worst_to_print} patients after registration:")
        for r in results_sorted[:n_worst_to_print]:
            print(f"  {r['patient_id']} {r['frame_id']} : dice_after={r['dice_after']:.3f} | gain={r['dice_gain']:+.3f}")

        results_ED = [r for r in results_all if r["frame_id"] in ("frame01", "frame04")]
        _print_dice_stats("ED frames only", results_ED)

        results_ES = [r for r in results_all if r["frame_id"] not in ("frame01", "frame04")]
        _print_dice_stats("ES frames only", results_ES)

        if registered_OLD is not None:
            new_reg_names = {
                Path(p).name.replace("_registered_gt.nii.gz", "")
                for p in glob.glob(str(PROCESSED_IMAGES_FOLDER / "registered_frames" / "patient*_frame*_registered_gt.nii.gz"))
            }
            old_reg_paths = sorted(glob.glob(
                str(PROCESSED_IMAGES_FOLDER / registered_OLD / "patient*_frame*_registered_gt.nii.gz")
            ))
            old_reg_names = {Path(p).name.replace("_registered_gt.nii.gz", "") for p in old_reg_paths}

            only_in_new = new_reg_names - old_reg_names
            only_in_old = old_reg_names - new_reg_names
            if only_in_new or only_in_old:
                print(f"\nWARNING: coverage mismatch between 'registered_frames' and '{registered_OLD}'")
                if only_in_new:
                    print(f"  Only in new ({len(only_in_new)}): {sorted(only_in_new)[:5]}")
                if only_in_old:
                    print(f"  Only in old ({len(only_in_old)}): {sorted(only_in_old)[:5]}")
            else:
                print(f"\nCoverage OK: both folders have {len(new_reg_names)} frames.")

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
