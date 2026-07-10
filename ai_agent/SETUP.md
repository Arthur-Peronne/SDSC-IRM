# SETUP.md — Operating the repository (read this first)

<!--
Stable onboarding for any agent (or person) operating this repo. Answers
"how does this project work and how do I run it safely?" Changes only when the
code changes — not per session. Session intent lives in program.md; hard rules
live in experiment.yaml.
-->

## What this project does
3D cardiac-MRI representation learning: preprocess images, reduce dimensionality
(spatial PCA, 3D autoencoders), then regress patient metadata on the latent
codes. The autonomous loop currently refines autoencoders to maximize downstream R².

## Environment
- Local: `.venv/`, `pip install -e .`. On Renku: Docker, no venv.
- Config is centralized (`src/config.py`, python-dotenv) and file-driven
  (`configs/*.yaml`). Imports use `from src.config import ...`; paths use
  `pathlib.Path`.
- Experiment tracking: MLflow via `src/tracking.py` (never call MLflow directly
  in scripts).

## Run one training manually (sanity check before autonomous mode)
```bash
python scripts/run_autoencoder.py        # reads configs/autoencoder.yaml
mlflow ui --backend-store-uri mlruns/    # inspect at http://127.0.0.1:5000
```

## The autonomous loop (what the agent operates)
- `ai_agent/program.md` — this session's research intent (human-written).
- `ai_agent/experiment.yaml` — the contract (mutable files, metric, budget).
- `ai_agent/driver.py` — the engine: lock → train → measure → decide → commit×2.
- `ai_agent/experiments/<hash>.md` — one record per trial.
- `ai_agent/trial_log.csv` — flat comparison index (maintained by the driver).

Run a trial: copy `experiments/TEMPLATE.md` to `experiments/draft.md`, fill
Hypothesis + Implementation + model_name, edit a mutable file, then:
```bash
python ai_agent/driver.py run <parent_id>
```
The driver commits the input, renames the draft to `<hash>.md`, runs N trainings
(per `repeat_over`), aggregates the metric, decides keep/revert, and commits.

## Mutable / frozen
You may ONLY modify files listed under `mutable` in experiment.yaml. NEVER touch anything outside the mutable allowlist in experiment.yaml.
The driver rejects any trial that modifies a file outside mutable. Everything not listed is frozen by default — this is the immutable judge that keeps results comparable.

## Reproducibility rules (project-critical)
- Never break past results. AE auto-commits go to branch `agent-ae-opti`, never `main`; review via the CSV, merge only what you trust.
- MLflow metadata (`mlruns/`) is committed for local `mlflow ui`; heavy artifacts (`.pth`, `.joblib`) are gitignored and stay in MLflow.
- Filenames are case-sensitive on Renku/Linux: write `program.md` (lowercase) consistently everywhere it is referenced.

## End of a campaign
A campaign = one fixed judge metric. max_trials defined in experiment.yaml. 
The current number of trials is calculated by counting rows in ai_agent/trial_log.csv (so the count is per-campaign only if the CSV is reset between campaigns).
