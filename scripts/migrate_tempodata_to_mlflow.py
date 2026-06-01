#!/usr/bin/env python3
"""
Migration: tempodata/autoencoder/ → MLflow

Run from project root:
    python scripts/migrate_tempodata_to_mlflow.py
    python scripts/migrate_tempodata_to_mlflow.py --dry-run
"""

import argparse
import re
from pathlib import Path

import mlflow

TEMPODATA_DIR = Path("tempodata/autoencoder")
MLFLOW_URI = "mlruns"
EXPERIMENT_NAME = "autoencoder"

RUN_FOLDER_RE = re.compile(
    r"^(?P<model>[A-Za-z0-9]+)_(?P<n_patients>\d+)patients"
    r"_split(?P<split_id>\d+)_(?P<latent_dim>\d+)dims$"
)


def parse_summarymetrics(path: Path) -> dict:
    metrics = {}
    current_metric = None
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.split(":")[0].strip() not in ("mean", "std", "min", "max", "median"):
            current_metric = line.rstrip(":").lower()
        else:
            key, value = line.split(":", 1)
            metrics[f"{current_metric}_{key.strip()}"] = float(value.strip())
    return metrics


def parse_loss_file(path: Path) -> tuple:
    lines = [l.strip() for l in path.read_text().splitlines() if l.strip()]

    if lines[0].startswith("best_epoch:"):
        best_epoch = int(lines[0].split(":")[1].strip())
        best_val_loss = float(lines[1].split(":")[1].strip())
        history = []
        for line in lines[2:]:
            m = re.match(r"train:\s*([\d.e+\-]+)\s+validation:\s*([\d.e+\-]+)", line)
            if m:
                history.append((float(m.group(1)), float(m.group(2))))
    else:
        best_epoch = int(lines[0].split(":")[1].strip())
        best_val_loss = None
        history = []
        for line in lines[1:]:
            m = re.match(r"train:\s*([\d.e+\-]+)", line)
            if m:
                history.append((float(m.group(1)), None))

    return best_epoch, best_val_loss, history


def migrate_tag_dir(tag_dir: Path, run_params: dict, tag: str, dry_run: bool) -> bool:
    # Need at least a validation or test summarymetrics
    has_metrics = (
        list(tag_dir.glob("*_summarymetrics_validation.txt"))
        or list(tag_dir.glob("*_summarymetrics_test.txt"))
    )
    if not has_metrics:
        print(f"    SKIP {tag} — no summarymetrics files")
        return False

    # Need a final loss file (not temp)
    loss_files = [f for f in tag_dir.glob("*_loss.txt") if "temp" not in f.name]
    if not loss_files:
        print(f"    SKIP {tag} — no loss file")
        return False

    best_epoch, best_val_loss, history = parse_loss_file(sorted(loss_files)[-1])

    run_name = (
        f"{run_params['model_name']}_{run_params['n_patients']}patients"
        f"_split{run_params['split_id']}_{run_params['latent_dim']}dims_{tag}"
    )
    val_loss_str = f"{best_val_loss:.6f}" if best_val_loss is not None else "N/A"
    print(f"    LOG  {run_name}  ({len(history)} epochs, best_val_loss={val_loss_str})")

    if dry_run:
        return True

    params = {**run_params, "experiment_tag": tag, "best_epoch": best_epoch}

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        if best_val_loss is not None:
            mlflow.log_metric("best_val_loss", best_val_loss)

        for step, (train_loss, val_loss) in enumerate(history, start=1):
            mlflow.log_metric("train_loss", train_loss, step=step)
            if val_loss is not None:
                mlflow.log_metric("val_loss", val_loss, step=step)

        for split in ("train", "validation", "test"):
            files = list(tag_dir.glob(f"*_summarymetrics_{split}.txt"))
            if files:
                for k, v in parse_summarymetrics(files[0]).items():
                    mlflow.log_metric(f"{split}_{k}", v)

        png = tag_dir / "train_val_loss.png"
        if png.exists():
            mlflow.log_artifact(str(png))

        for pth in tag_dir.glob("*.pth"):
            if "temp" not in pth.name:
                mlflow.log_artifact(str(pth))

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Parse only, no MLflow writes")
    args = parser.parse_args()

    if args.dry_run:
        print("=== DRY RUN — no MLflow writes ===\n")

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    migrated, skipped = 0, 0

    for run_dir in sorted(TEMPODATA_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        m = RUN_FOLDER_RE.match(run_dir.name)
        if not m:
            print(f"SKIP {run_dir.name} — folder name doesn't match pattern")
            continue

        run_params = {
            "model_name": m.group("model"),
            "n_patients": m.group("n_patients"),
            "split_id": m.group("split_id"),
            "latent_dim": m.group("latent_dim"),
        }
        print(f"\n{run_dir.name}")

        for tag_dir in sorted(run_dir.iterdir()):
            if not tag_dir.is_dir():
                continue
            ok = migrate_tag_dir(tag_dir, run_params, tag_dir.name, args.dry_run)
            if ok:
                migrated += 1
            else:
                skipped += 1

    print(f"\n=== Done: {migrated} runs migrated, {skipped} skipped ===")


if __name__ == "__main__":
    main()