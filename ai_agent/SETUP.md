<!--
Stable onboarding for any agent (or person) operating this repo. Answers
"how does this project work and how do I run it safely?" Changes only when the
code changes — not per session. Session intent lives in program.md; hard rules
live in experiment.yaml.
-->

# SETUP.md — Operating the repository (read this first)

## What this project does
3D cardiac-MRI representation learning: preprocess images, reduce dimensionality
(spatial PCA, 3D autoencoders), then regress patient metadata on the latent
codes. 
The autonomous loop currently refines autoencoders to maximize downstream
classification accuracy (patient group, logistic regression on the AE latent codes)
— not autoencoder R² directly, which is still logged for context but no longer
the judge.

## Environment
- Local: `.venv/`, `pip install -e .`. On Renku: Docker, no venv (pre-activated shell).
- Config is centralized (`src/config.py`, python-dotenv) and file-driven
  (`configs/*.yaml`). Imports use `from src.config import ...`; paths use `pathlib.Path`.
- Experiment tracking: MLflow via `src/tracking.py` (scripts never call MLflow directly).
- All commands are run from the repo root.

## Run one training manually (sanity check before autonomous mode)
```bash
python scripts/run_autoencoder.py        # reads configs/autoencoder.yaml
mlflow ui --backend-store-uri mlruns/    # inspect at http://127.0.0.1:5000
```
Run by hand, the script ignores the driver-only flags (`--set`, `--trial-id`) and
produces an untagged run — exactly the classic behaviour.

To sanity-check the classification phase alone:
```bash
python scripts/run_regression.py         # reads configs/regression.yaml
```
Run by hand, the script ignores the driver-only flags (`--trial-id`, `--ae-trial-tag`,
`--ae-filter`) and finds its AE run via the manual filters in `regression.yaml`
(`model_name`/`experiment_tag`/`split_name`) instead of by tag.

## The autonomous loop (what the agent operates)
- `ai_agent/program.md` — this session's research intent (human-written).
- `ai_agent/experiment.yaml` — the contract: mutable files, metric, verdict rule, budget.
- `ai_agent/driver.py` — the engine: lock → train → measure → decide → commit×2.
- `ai_agent/experiments/<id>.md` — one record per trial (`<id>` = short sha of commit 1).
- `ai_agent/trial_log.csv` — flat comparison index (owned by the driver; never edit by hand).

Run a trial: create the draft by copying the template
(`cp ai_agent/experiments/TEMPLATE.md ai_agent/experiments/draft.md`), fill
`model_name` / `summary` / `parent` and the Hypothesis + Implementation prose,
edit a mutable file, then:
```bash
python ai_agent/driver.py run
```
No arguments: lineage (`parent`) is declared in the draft's frontmatter, not on the
command line. The driver commits the input (its short sha becomes the trial `id`),
renames the draft to `<id>.md`, then runs two phases per `repeat_over` value:
1. N autoencoder trainings (`scripts/run_autoencoder.py`), each tagged `trial_id=<id>_ae`.
2. N classification runs (`scripts/run_regression.py`), each tagged `trial_id=<id>_clf`
   and pointing back at its matching AE run via `--ae-trial-tag`/`--ae-filter` (the
   driver never touches configs/regression.yaml — it passes these as CLI flags, same
   spirit as `--set` for the AE phase).

The driver reads the `_clf` runs back BY TAG, aggregates, decides the verdict, and
commits the result. `_ae` and `_clf` run IDs are both recorded in the trial's `.md`,
kept separate.

## Two axes on each record (kept separate)
- `status`  — lifecycle, lowercase:  `draft` → `completed` | `failed`.
  `failed` = mechanical (crash, NaN, wrong run count); no comparable metric produced.
- `verdict` — judgement, UPPERCASE:  `BASELINE` | `CHAMPION` | `CANDIDATE` | `FAILURE`.
  `FAILURE` = the run completed fine but did not beat the champion.
Both `failed` and `FAILURE` revert the mutable code; only kept verdicts (BASELINE /
CHAMPION / CANDIDATE) leave it in place.

## Mutable / frozen
You may ONLY modify files listed under `mutable` in experiment.yaml. The driver
rejects (before committing or training) any trial that changes anything else —
including experiment.yaml, driver.py and program.md, which are frozen too. The only
writable area outside `mutable` is the trial-record dir (`experiments/`). Everything
frozen is the immutable judge that keeps results comparable.

## Config prerequisites for a campaign (avoid a silent `failed`)
The metric the driver reads now comes from the classification phase
(`configs/regression.yaml`), not directly from the autoencoder:
- **`configs/autoencoder.yaml` and `configs/regression.yaml` must have IDENTICAL
  `n_train` / `n_val` / `n_test` / `special_split` / `stratify_ongroup`.** A mismatch
  is caught by `_verify_split` and fails the trial cleanly — not silently wrong, but
  the two files are never synced automatically; check both by hand before a new
  campaign.
- **`regression.yaml`'s `eval_on: "val"` is what makes the judge use the validation
  set** — the 30-patient test set is never touched by the agent, reserved for a
  one-time final AE-vs-PCA comparison at the end of the campaign (`eval_on: "test"`).
- `n_val > 0` in both files (needed for AE early stopping AND as the classifier's
  judge set when `eval_on: "val"`).
- **Manual hyperparameter tuning requires `hyper_automatic_values: false`**, otherwise
  the hyperparameters the agent writes in the YAML are ignored and re-resolved from
  `configs/ae_HPforarchis.yaml`.
- Keep `multiple_models_and_dims: false`: the driver owns the sweep via `repeat_over`.
- `seed`: set an integer for reproducible runs; `null` for a random (non-reproducible) run.

## Reproducibility rules (project-critical)
- Never break past results. Auto-commits go to the isolated branch defined in
  experiment.yaml (`decision.branch`), never `main`; review via the CSV, merge only
  what you trust.
- Any trial `id` is a git short sha: `git show <id>` shows the exact frozen input
  (code + all configs + experiment.yaml) that produced it.
- MLflow metadata (`mlruns/`) is committed for local `mlflow ui`; heavy artifacts
  (`.pth`, `.joblib`) are gitignored and stay on disk only.
- Filenames are case-sensitive on Renku/Linux: write `program.md` (lowercase) everywhere.

## End of a campaign
A campaign = one fixed judge metric + a budget (`max_trials` in experiment.yaml).
`max_trials` counts ALL trials in `trial_log.csv`, failures included; the count only
resets when the CSV is archived and emptied. To start a new campaign: archive the CSV
and the per-trial records, then edit experiment.yaml (new `mutable` / metric / budget).