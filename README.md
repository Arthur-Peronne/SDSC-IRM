# SDSC-IRM

Representation learning for cardiac cine-MRI on the ACDC dataset: dimensionality
reduction (PCA, autoencoders), image registration, and patient-group
classification, with every stage documented and reproducible.

## Overview

The heart is imaged over time with cine-MRI, giving a 4D volume `(x, y, z, t)`
per patient. This project studies how to **compress** these volumes into a small
latent representation and what that representation is good for — reconstructing
held-out patients and predicting their diagnostic group.

The data is the **ACDC** dataset (Bernard et al., 2018): 150 patients, evenly
split across five groups — dilated cardiomyopathy (DCM), hypertrophic
cardiomyopathy (HCM), myocardial infarction (MINF), abnormal right ventricle
(RV), and normal (NOR) — each with expert segmentations at the end-diastole (ED)
and end-systole (ES) frames.

The pipeline runs in stages: visualize → temporal PCA → registration → spatial
PCA → classification → autoencoders → regularization → VAE → automatic
architecture search. Each stage is a script driven by a YAML config, tracked with
MLflow, and written up in `reports/`.

## Scientific summary

The project builds up from a simple linear baseline to non-linear models, using a
consistent train / validation / test split and reconstruction R² (on held-out
patients) and group-classification accuracy as the two comparison metrics.

- **Temporal PCA** — per patient, the cardiac cycle is low-rank: a handful of
  components explain most of the variance and trace the cycle in the eigenbasis.
- **Registration** — a resample → crop → rigid-registration pipeline brings every
  patient onto a common voxel grid, so a voxel means the same anatomical location
  across patients (a prerequisite for the across-patient models).
- **Spatial PCA** — PCA across patients yields interpretable anatomical modes and
  a strong linear reconstruction baseline that generalizes to held-out patients.
- **Classification** — off-the-shelf classifiers (logistic regression, random
  forest, XGBoost) predict the ACDC group from the latent space. RV and DCM are
  well separated; NOR, MINF and HCM are frequently confused.
- **Autoencoders** — three hand-designed 3D convolutional architectures are
  compared as a non-linear alternative to spatial PCA.
- **Regularization & hyperparameter optimization** — weight decay, dropout and
  denoising, combined with an Optuna hyperparameter search, studied as a function
  of latent-space size.
- **Variational autoencoder** — a probabilistic latent variant (β-VAE with KL
  warm-up), with latent-space analysis.
- **Agentic architecture search** — an LLM-driven agent proposes, implements and
  evaluates new encoder architectures and hyperparameters, in two campaigns — one
  optimizing reconstruction R², one optimizing classification accuracy. The loop
  is **semi-deterministic and fully logged** (per-trial inputs and outputs frozen
  in Git, automatic revert on failed trials, one MLflow run per trial) for
  stability and reproducibility.

### Conclusions

Group-classification accuracy is remarkably stable across choices that ought to
matter. Logistic regression, random forest and XGBoost land within noise of one
another (report 04), and so do very different encoders — linear PCA even matches
or slightly beats the convolutional autoencoders at equal latent dimension
(reports 03–05). The limiting factor is therefore the **representation**, not the
classifier or the encoder architecture.

The latent dimension gives a clean compression story. Reconstruction quality
rises steeply with the first few components and then plateaus — around R² ≈ 0.8
for the autoencoders, below PCA's ≈ 0.9 on the development set — while
classification accuracy saturates by about **20 dimensions**. Roughly twenty
numbers per heart are thus close to optimal for both tasks; beyond that, extra
capacity mostly adds parameters to regularize rather than signal to exploit, and
larger latent spaces indeed require more regularization (weight decay, dropout) to
avoid overfitting (report 06).

Two open questions bound these results. First, registration is only rigid and
reaches a mean Dice of ≈ 0.72, so inter-patient alignment is imperfect; the
residual misalignment — and, more broadly, the quality and size of the
150-patient dataset — may cap how much structure any representation can recover.
Second, the accuracy ceiling (≈ 0.62–0.65) sits well below the near-perfect
numbers reported on ACDC by Bernard et al. (2018): a natural next direction is to
move toward their approach, deriving features from the expert segmentations rather
than learning them from the raw voxels.

### Reports

Detailed write-ups — objective, method, config, results, and figures — live in
[`reports/`](reports/):

- [`00_mri_visualization`](reports/00_mri_visualization/) — plotting raw and processed MRI
- [`01_pca_temporal`](reports/01_pca_temporal/) — per-patient temporal PCA
- [`02_registration`](reports/02_registration/) — resample / crop / rigid registration
- [`03_pca_spatial`](reports/03_pca_spatial/) — across-patient spatial PCA
- [`04_classification`](reports/04_classification/) — group classification (logistic / RF / XGBoost)
- `05_autoencoders_architectures`— the three base 3D AE architectures
- `06_regularization` — regularization and Optuna hyperparameter search
- `07_vae` — variational autoencoder and latent-space analysis
- `08_ai_agent_r2` — AI-agent architecture search optimizing reconstruction R²
- `09_ai_agent_classification` — AI-agent architecture search optimizing accuracy

The reports 05–09 will be finalized mid-August.

## Repository structure

Each pipeline stage is one script in `scripts/` that reads its config from
`configs/` and logs its run to MLflow. The reusable logic lives in `src/`.

```
SDSC-IRM/
├── src/
│   ├── config.py            # resolves paths from .env
│   ├── tracking.py          # MLflow helpers
│   ├── data/                # import, resampling, geometry, registration, loader, splits
│   ├── models/              # pca, pca_spatial, ae_models, regression
│   ├── training/            # autoencoder training loop
│   └── visualization/       # mri_plots, pca_plots, regression_plots
├── scripts/                 # one run_*.py per stage
│   ├── run_visualize.py
│   ├── run_pca_temporal.py
│   ├── run_registration.py
│   ├── run_pca_spatial.py
│   ├── run_regression.py
│   ├── run_autoencoder.py
│   ├── run_ae_optuna.py
│   └── run_comparison_aepca.py
├── configs/                 # one YAML per script (hyperparameters, split, I/O)
├── reports/                 # per-stage write-ups (report + config_files/ + figures/)
├── ai_agent/                # architecture-search driver (reports 08–09)
├── mlruns/                  # MLflow store (metadata committed for local UI)
├── requirements_locked.txt
├── pyproject.toml
└── .env.example
```

Preprocessing outputs (resampled / cropped / registered frames, cached arrays) go
to the folders set in `.env`; model runs, metrics and artifacts are tracked in
`mlruns/`.

## Setup

Follow these steps in order.

### 1. Environment variables

Create a `.env` file from the template and update the paths for your machine:

```bash
cp .env.example .env
```

It defines two groups of variables:

- **Paths** — `DATADIR` (raw ACDC tree), `PROCESSED_IMAGES_FOLDER` (preprocessed
  frames), `RESULTS_FOLDER` (figures and metrics), and `MLRUNS_FOLDER` (MLflow
  store). The raw ACDC data is **not** included in the repository and must be
  available at `DATADIR`.
- **Object-storage credentials** — the data is hosted on Switch SWITCHengines S3
  object storage (region ZH). Fill in `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`
  and `ENDPOINT_URL` (obtained from engines.switch.ch → Project → API Access) to
  access the bucket directly; the two `*_CHECKSUM_*` settings avoid checksum
  errors with this backend. On a Renku session the bucket is mounted as a data
  connector, so these are only needed when accessing S3 outside that mount.

### 2. Virtual environment (skip on a standard writable Renku session)

On your own machine, create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies and the project

Install the locked requirements and the project itself, then run scripts from the
project root:

```bash
pip install -r requirements_locked.txt
pip install -e .
```

A typical run edits the relevant config and launches its script, e.g.:

```bash
python scripts/run_pca_spatial.py      # reads configs/pca_spatial.yaml
```

## Useful commands

```bash
# MLflow UI (from the project root, venv activated)
mlflow ui --backend-store-uri mlruns/
# → http://127.0.0.1:5000
```

```bash
# Launch a long job in the background and log its output
nohup python -u scripts/run_autoencoder.py > run_autoencoder.log 2>&1 &
```

```bash
# Create archives to move data between Renku sessions
tar -czf tempdata_autoencoder.tar.gz tempodata/autoencoder/
tar -czf ACDC.tar.gz /home/renku/work/s3-bucket/ACDC
tar -czf results.tar.gz results/
```

## References

O. Bernard, A. Lalande, C. Zotti, F. Cervenansky, et al., "Deep Learning
Techniques for Automatic MRI Cardiac Multi-structures Segmentation and Diagnosis:
Is the Problem Solved?", *IEEE Transactions on Medical Imaging*, vol. 37, no. 11,
pp. 2514–2525, Nov. 2018. doi: 10.1109/TMI.2018.2837502