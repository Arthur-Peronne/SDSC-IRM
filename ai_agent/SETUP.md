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
codes. The autonomous loop currently refines autoencoders to maximize validation R².

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

## The autonomous loop (what the agent operates)
- `ai_agent/program.md` — this session's research intent (human-written).
- `ai_agent/experiment.yaml` — the contract: mutable files, metric, verdict rule, budget.
- `ai_agent/driver.py` — the engine: lock → train → measure → decide → commit×2.
- `ai_agent/experiments/<id>.md` — one record per trial (`<id>` = short sha of commit 1).
- `ai_agent/trial_log.csv` — flat comparison index (owned by the driver; never edit by hand).

Run a trial: copy `experiments/TEMPLATE.md` to `experiments/draft.md`, fill
`model_name` / `summary` / `parent` and the Hypothesis + Implementation prose,
edit a mutable file, then:
```bash
python ai_agent/driver.py run
```
No arguments: lineage (`parent`) is declared in the draft's frontmatter, not on the
command line. The driver commits the input (its short sha becomes the trial `id`),
renames the draft to `<id>.md`, runs N trainings (per `repeat_over`, each tagged
`trial_id=<id>` in MLflow), reads those runs back BY TAG, aggregates, decides the
verdict, and commits the result.

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
The metric the driver reads depends on how `configs/autoencoder.yaml` is set:
- **`validation_R2_mean` / `validation_MSE_mean` require `n_val > 0` AND
  `compute_metrics: true`** — they are computed on the validation set. With `n_val: 0`
  they are never logged and the trial fails when the driver cannot find the metric.
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

Then purge the FAILURE and `failed` runs from MLflow (their `<id>.md` + CSV row remain
as the trace; only the models leave). The `trial_id` tag identifies them — no run IDs
to copy by hand. Run from the repo root, ONLY when no trial is running:

```bash
python3 - <<'PY'
import csv, subprocess, pathlib
bad = {r["id"] for r in csv.DictReader(open("ai_agent/trial_log.csv"))
       if r["verdict"] == "FAILURE" or r["status"] == "failed"}
for tagfile in pathlib.Path("mlruns").glob("*/*/tags/trial_id"):
    if tagfile.read_text().strip() in bad:
        run_dir = tagfile.parent.parent          # mlruns/<experiment_id>/<run_id>
        print("purging", run_dir)
        subprocess.run(["rm", "-rf", str(run_dir)])
PY
git add -A mlruns/
git commit -m "cleanup: purge FAILURE/failed runs (end of campaign)"
```
This removes only the run directories identified by tag (committed metadata AND
gitignored `.pth`). NEVER run `rm -rf mlruns/` or `git clean -fdx mlruns/` globally —
that would wipe CHAMPION and CANDIDATE models too.