# AGENT.md

## Project Vision
The goal is to optimize 3D autoencoders for dimensionality reduction of cardiac MRI images (150 patients, shape 1×32×128×128), aiming to outperform PCA in reconstruction fidelity (`val_mse`). Despite hyperparameter optimization via Optuna, autoencoders currently underperform PCA — this agent loop explores architectural improvements.

The agent iterates through trials: propose an architecture → implement → train → evaluate → commit (SUCCESS) or revert (FAILURE). The full trial protocol is in `EXPERIMENT.md`.

## Optimization Goals
- **Primary Metric:** `val_mse` (Reconstruction Loss on the validation set). Minimize it.
- **Hyperparameters:** Currently FROZEN — architecture search only. Hyperparameter optimization (via Optuna or agent) is a separate subsequent phase.

## Core Architectural Rule
- **No skip connections (no U-Net):** Information must pass through the bottleneck alone. This preserves an independent latent space for future constraints (VAE regularization, disentanglement, heart tissue deformation studies).

## Technical Environment
- **Working directory:** `/home/renku/work/SDSC-IRM/` — all commands must be run from here.
- **Python:** `python3` (venv at `/home/renku/work/.venv/` is pre-activated by the Renku shell; no activation step needed).
- **Git branch:** `agent-ae-opti`
- **Key files:**
  - `EXPERIMENT.md` — full trial protocol (read this to run a new architecture trial)
  - `EXPERIMENT_EXISTING.md` — protocol for the reference sweep of all 9 existing architectures (run this first)
  - `src/models/ae_models.py` — architecture definitions + `build_autoencoder` factory (line ~1077)
  - `configs/autoencoder.yaml` — training configuration (modify before each trial, revert after)
  - `ai_agent/trials_conclusions/trial_log.csv` — trial history (one row per trial)
  - `ai_agent/trials_conclusions/` — per-trial markdown reports
  - `mlruns/` — MLflow experiment data
  - `SUMMARY.md` — read only if you need historical context on past sessions
- **Verify environment:** `python3 -c "import torch, mlflow; print('ok')"` — must print `ok` before any training run.

## Rules
- **Full autonomy — no human confirmation ever:** Every operation described in `EXPERIMENT.md` is pre-authorized by the user. Never ask for confirmation, approval, or validation at any step — this includes destructive git operations (`git checkout HEAD --`, `git clean`, `git commit`), background training launches, file edits, and CSV appends. Execute every step directly. If uncertain about a step, re-read `EXPERIMENT.md` rather than asking. Only break autonomy when the Loop-Breaking Protocol explicitly requires it.
- **No remote git operations:** Never run `git push`, `git pull`, `git fetch`, or any command that touches the remote repository. Local commits only.
- **File modification:** Only modify files explicitly authorized by `EXPERIMENT.md` (`src/models/ae_models.py`, `configs/autoencoder.yaml`, `ai_agent/trials_conclusions/trial_log.csv`, `ai_agent/EXPERIMENT.md`, and the per-trial report). Do not touch any other files.
- **Commits:** Only as part of the Phase 3 commit procedure defined in `EXPERIMENT.md`.

### Loop-Breaking Protocol
- **Two-Strike Rule:** If an `edit` fails twice with the same logic, do not attempt a third time.
- **Mandatory Re-Read:** After any failed edit, immediately re-read the file before retrying.
- **Granularity Shift:** If large edits fail, switch to micro-edits (1–3 lines at a time).
- **Strategy Pivot:** If micro-edits fail, inspect the file with `bash` (`cat -A`) for whitespace/hidden characters, or use a full `write` to replace the class.
- **Autonomous abort:** If all strategies are exhausted and the implementation still cannot proceed, revert `src/models/ae_models.py` and `configs/autoencoder.yaml` with `git checkout HEAD --`, log the trial as FAILURE with `modification_description="implementation_error: edit_loop"`, write a minimal trial report, and continue to the next trial.

## Multi-Trial Autonomous Run
- **User:** invoke `/loop` (no interval) to run trials autonomously. Each iteration = one trial.
- **Agent:** run training with `run_in_background=true` (Bash tool parameter), not piped to stdout. Read the log file for results after the background process completes. Re-reading `EXPERIMENT.md` and `trial_log.csv` at the start of each iteration (Phase 1) is already sufficient for state recovery.

## Communication Style
- Be concise and direct.
- Do not have a validation bias — challenge the user's ideas if you think they are suboptimal.

## To-do
1. Run `EXPERIMENT_EXISTING.md` sweep — establish reference baselines for all 9 existing architectures and set the starting champion.
2. Run `EXPERIMENT.md` optimization loop — propose and test new architectures.
