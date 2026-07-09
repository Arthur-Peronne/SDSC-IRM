#!/usr/bin/env python
"""
driver.py -- the deterministic engine of the autonomous research loop.

The AGENT edits code and writes prose; the DRIVER does the mechanical,
incorruptible part: lock the input, train, read the metric, decide keep/revert
by a fixed numeric rule, and commit. The agent and the driver NEVER run at the
same time -- they take turns. The driver takes NO "intelligent" decision.

Turn structure (one trial):
  [agent]  copies experiments/TEMPLATE.md -> experiments/draft.md, edits a file
           in `mutable`, writes Hypothesis + Implementation + model_name.
  [driver] python driver.py run [parent_id]   <-- everything below happens here
             1. scope check (only `mutable` files changed?)
             2. commit 1 -> freezes input (code + all configs + experiment.yaml),
                yields commit sha -> computes id -> renames draft.md to <id>.md
             3. run eval: N trainings over repeat_over (N=1 if null), fixed epochs
             4. read per-run metrics from MLflow -> aggregate into trial scalar
             5. compare to champion -> keep or git revert  (deterministic)
             6. write ## Results into <id>.md + append CSV row
             7. commit 2 -> freezes output
  [agent]  reads result, writes Training Dynamics + Conclusion.

SKELETON: functions that touch project specifics (MLflow field names, git
plumbing, how the training script receives repeat overrides) are marked TODO.
Everything else is driven by experiment.yaml, so the same driver works on any
project -- a user edits only the YAML + program.md.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

CONTRACT_PATH = Path("ai_agent/experiment.yaml")
DRAFT_NAME = "draft.md"          # working name before the hash exists


# -----------------------------------------------------------------------------
# Contract loading
# -----------------------------------------------------------------------------
def load_contract(path: Path = CONTRACT_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# -----------------------------------------------------------------------------
# 1. Scope check -- ALLOWLIST enforcement (deny by default)
# -----------------------------------------------------------------------------
def assert_only_mutable_changed(mutable: list[str]) -> None:
    changed = _git_changed_files()
    # the draft record itself is allowed to change (agent wrote into it)
    allowed = set(mutable)
    illegal = [f for f in changed
               if f not in allowed and not f.startswith("ai_agent/")]
    if illegal:
        raise ScopeViolation(
            f"Agent modified frozen files: {illegal}. "
            f"Only {mutable} are mutable. Trial rejected."
        )


def _git_changed_files() -> list[str]:
    out = subprocess.run(["git", "diff", "--name-only", "HEAD"],
                         capture_output=True, text=True, check=True).stdout
    return [ln.strip() for ln in out.splitlines() if ln.strip()]


# -----------------------------------------------------------------------------
# 2. Identity + hash   id = sha256(canonical_json(identity))[:12]
# -----------------------------------------------------------------------------
def compute_id(identity: dict) -> str:
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def build_identity(commit_sha: str, command: str, parent: str | None) -> dict:
    # `command` is the DRIVER call (not the N sub-trainings): the repeat_over
    # that defines the N runs lives in experiment.yaml, itself captured by the
    # input commit. So commit + this command fully pin the trial.
    # `parent` is lineage metadata only -- for a merged-idea trial it points to
    # the code we branched FROM; the fusion is described in ## Hypothesis.
    return {"parent": parent, "commit": commit_sha, "command": command}


# -----------------------------------------------------------------------------
# Frontmatter read/write for the <hash>.md record
# -----------------------------------------------------------------------------
def read_frontmatter(md_path: Path) -> dict:
    text = md_path.read_text()
    m = re.search(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    if not m:
        raise ValueError(f"No frontmatter found in {md_path}")
    return yaml.safe_load(m.group(1)) or {}


# -----------------------------------------------------------------------------
# 3. Run the eval: N trainings over `repeat_over` (N=1 if null)
# -----------------------------------------------------------------------------
def _repeat_values(cfg: dict) -> list[dict]:
    """Per-run overrides. [{}] means a single run (N=1)."""
    ro = cfg["eval"].get("repeat_over")
    if not ro:
        return [{}]
    (axis, values), = ro.items()                 # e.g. ("latent_dim", [8,60,240])
    return [{axis: v} for v in values]


def run_eval(cfg: dict, log_path: Path) -> None:
    base_cmd = cfg["eval"]["command"].split()
    with open(log_path, "w") as logf:
        for override in _repeat_values(cfg):
            cmd = list(base_cmd)
            # TODO: pass `override` to your script (e.g. --latent_dim 8 / --seed 0)
            #   for k, v in override.items(): cmd += [f"--{k}", str(v)]
            # Ensure run_budget.n_epochs is the same ceiling for every run and
            # early stopping is configured identically (in autoencoder.yaml).
            logf.write(f"\n=== run override={override} ===\n"); logf.flush()
            proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
            if proc.returncode != 0:
                raise EvalFailed(f"Training failed for override={override} (see {log_path})")


# -----------------------------------------------------------------------------
# 4. Read per-run metrics from MLflow, then AGGREGATE into the trial scalar.
#    The aggregate is TRIAL-level: it goes to <hash>.md, never to MLflow.
# -----------------------------------------------------------------------------
def read_primary_metric(cfg: dict) -> float:
    per_run = cfg["eval"]["per_run_metric"]
    n = len(_repeat_values(cfg))                 # 1 for a single training
    values = _read_recent_run_metrics(per_run, n)
    if len(values) != n:
        raise EvalFailed(f"Expected {n} runs with '{per_run}', found {len(values)}")
    return _aggregate(values, cfg["eval"]["aggregation"])


def _aggregate(values: list[float], how: str) -> float:
    if how == "identity":                        # N=1: the value itself
        return values[0]
    if how == "mean":                            # N>1: average over the axis
        return sum(values) / len(values)
    raise ValueError(f"Unknown aggregation '{how}'. Use 'identity' or 'mean'.")
    # (std for seed-averaging will be added here when that campaign starts.)


def _read_recent_run_metrics(metric_name: str, n: int) -> list[float]:
    """The n most recent runs' `metric_name` from MLflow.
    TODO (robustness): tag this trial's runs with the trial id at launch and
    filter on that tag instead of recency -- avoids averaging the wrong runs if
    the same config is relaunched. Sketch with recency:
        import mlflow
        runs = mlflow.search_runs(experiment_names=["autoencoder"],
                                  order_by=["start_time DESC"], max_results=n)
        return [float(runs.iloc[i][f"metrics.{metric_name}"]) for i in range(n)]
    """
    raise NotImplementedError(f"Wire MLflow read for '{metric_name}' (n={n})")


# -----------------------------------------------------------------------------
# 5. Deterministic keep/revert  -- NO LLM here
# -----------------------------------------------------------------------------
def is_better(candidate: float, champion: float | None, direction: str) -> bool:
    if champion is None:                         # first trial becomes baseline
        return True
    return candidate > champion if direction == "maximize" else candidate < champion


def read_champion_metric(ledger: Path) -> float | None:
    if not ledger.exists():
        return None
    best = None
    with open(ledger) as f:
        for row in csv.DictReader(f):
            if row.get("decision") == "keep":
                v = float(row["metric_value"])
                best = v if best is None else max(best, v)
    return best

def trial_count(ledger: Path) -> int:
    """Number of trials in the current campaign = data rows in the CSV (header
    excluded). Resets to 0 when the CSV is archived+emptied between campaigns."""
    if not ledger.exists():
        return 0
    with open(ledger) as f:
        return sum(1 for _ in csv.DictReader(f))

# -----------------------------------------------------------------------------
# 6/7. Report + ledger + commits
# -----------------------------------------------------------------------------
def write_results_section(md_path: Path, metric: float, delta, status: str) -> None:
    """Fill ## Results and update the STATUS in the title of the <hash>.md.
    delta is DISPLAY-ONLY (recomputed for readability, goes stale when champion
    changes) -- it is never used in the decision.
    TODO: populate the ## Results block per TEMPLATE.md layout
    (R2 per dim, MLflow run ids, best epochs) and set the frontmatter fields
    id/status/metric.primary.value/created_at.
    """
    ...


def append_ledger_row(ledger: Path, row: dict) -> None:
    """Append one row. The driver is the sole owner of the header: it is written
    automatically on the first trial (when the file does not yet exist) and
    skipped afterwards. Do NOT create this file by hand."""
    header = ["timestamp", "id", "parent", "model_name", "modification_description",
              "metric_name", "metric_value", "metric_delta", "decision", "status"]
    exists = ledger.exists()
    with open(ledger, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists:
            w.writeheader()
        w.writerow(row)


def git_commit(msg: str, branch: str) -> str:
    subprocess.run(["git", "checkout", "-q", branch], check=False)   # ensure branch
    subprocess.run(["git", "add", "-A"], check=True)
    subprocess.run(["git", "commit", "-q", "-m", msg], check=True)
    return subprocess.run(["git", "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()


def git_revert_worktree(mutable: list[str]) -> None:
    """Discard the code change of a rejected trial (keep the .md record)."""
    # revert only the mutable files, preserving the experiment record under ai_agent/
    for path in mutable:
        subprocess.run(["git", "checkout", "--", path], check=False)


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------
def run_trial(parent: str | None = None) -> None:
    cfg = load_contract()
    exp_dir = Path(cfg["logging"]["experiments_dir"])
    ledger = Path(cfg["logging"]["ledger"])
    branch = cfg["decision"]["branch"]
    mutable = cfg["mutable"]

    draft = exp_dir / DRAFT_NAME
    if not draft.exists():
        raise FileNotFoundError(
            f"{draft} not found. Copy {cfg['logging']['template']} to {draft} "
            "and fill Hypothesis / Implementation / model_name before running."
        )

    # 0. campaign cap — refuse before doing any work (no commit, no training)
    max_trials = cfg["decision"].get("max_trials")
    if max_trials is not None and trial_count(ledger) >= max_trials:
        print(f"Reached max_trials={max_trials} for this campaign. "
              "Archive trial_log.csv + reports and start a new campaign to continue.")
        return

    # 1. scope
    assert_only_mutable_changed(mutable)

    # 2. commit 1 (input) -> identity -> id -> rename draft to <id>.md
    driver_cmd = "python ai_agent/driver.py run" + (f" {parent}" if parent else "")
    pre_sha = git_commit("[trial] lock input", branch)
    identity = build_identity(pre_sha, driver_cmd, parent)
    trial_id = compute_id(identity)
    md_path = exp_dir / f"{trial_id}.md"
    if md_path.exists():
        raise ValueError(f"Duplicate trial id {trial_id} (identical identity). Aborting.")
    draft.rename(md_path)                          # draft.md -> <id>.md

    # read agent-provided descriptive fields from the frontmatter
    fm = read_frontmatter(md_path)
    model_name = fm.get("model_name", "")

    # 3. train (N runs over repeat_over)
    log_path = exp_dir / f"{trial_id}.console.log"
    run_eval(cfg, log_path)

    # 4. read + aggregate metric
    metric = read_primary_metric(cfg)
    champion = read_champion_metric(ledger)
    # delta is DISPLAY-ONLY, never decisional
    delta = metric - champion if champion is not None else 0.0

    # 5. decide (deterministic)
    keep = is_better(metric, champion, cfg["eval"]["direction"])
    status = ("BASELINE" if keep and champion is None
              else "CHAMPION" if keep else "FAILURE")
    if not keep:
        git_revert_worktree(mutable)

    # 6. write results + ledger row (model_name comes from the .md frontmatter)
    write_results_section(md_path, metric, delta, status)
    append_ledger_row(ledger, {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "id": trial_id, "parent": parent or "",
        "model_name": model_name,
        "modification_description": fm.get("summary", ""),
        "metric_name": cfg["eval"]["primary_metric_name"],
        "metric_value": metric, "metric_delta": delta,
        "decision": "keep" if keep else "revert", "status": status,
    })

    # 7. commit 2 (output)
    git_commit(f"[trial] {trial_id} result: {status} ({metric:.4f})", branch)
    print(f"Trial {trial_id}: {status}  {cfg['eval']['primary_metric_name']}={metric:.4f}  delta={delta:+.4f}")


class ScopeViolation(Exception): ...
class EvalFailed(Exception): ...


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "run":
        parent = sys.argv[2] if len(sys.argv) > 2 else None
        run_trial(parent=parent)
    else:
        print(f"Unknown command: {cmd}. Use: python driver.py run [parent_id]")
        sys.exit(1)