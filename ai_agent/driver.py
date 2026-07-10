#!/usr/bin/env python
"""
driver.py -- the deterministic engine of the autonomous research loop.

The AGENT edits code and writes prose; the DRIVER does the mechanical,
incorruptible part: lock the input, train, read the metric, decide the verdict
by a fixed numeric rule, and commit. Agent and driver take turns; the driver
takes NO "intelligent" decision.

Turn structure (one trial):
  [agent]  copies experiments/TEMPLATE.md -> experiments/draft.md, edits a file
           in `mutable`, fills Hypothesis / Implementation / model_name / parent.
  [driver] python ai_agent/driver.py run       <-- everything below happens here
             0. campaign cap (refuse if ledger already has max_trials rows)
             1. scope check: only `mutable` files + the trial record changed?
             2. commit 1 -> freezes input (code + all configs + experiment.yaml);
                the commit's short sha IS the trial id; draft.md -> <id>.md
             3. run eval: N trainings over repeat_over (N=1 if null), each run
                tagged in MLflow with trial_id. ALL-OR-NOTHING: any failing run
                fails the whole trial.
             4. read the N runs back BY TAG, aggregate -> one decisional scalar
             5. compare to champion -> BASELINE / CHAMPION / CANDIDATE / FAILURE
             6. write ## Results + frontmatter into <id>.md, append CSV row
             7. commit 2 -> freezes output
  [agent]  reads result, writes Training Dynamics + Conclusion.

Two axes, kept separate (never conflated):
  status  (lifecycle, lowercase) : draft -> completed | failed
  verdict (judgement, UPPERCASE) : BASELINE | CHAMPION | CANDIDATE | FAILURE
"failed" = a MECHANICAL failure (crash, NaN, wrong run count); the run never
produced a comparable metric. "FAILURE" = the run completed fine but lost to the
champion. Both revert the mutable code; only the reason and the status differ.

The training script must expose two hooks (see the two `HOOK` comments below):
  --trial-id <id>     set an MLflow tag  trial_id=<id>  on the run
  --set key=value     override cfg[key] in memory (the on-disk YAML is untouched)
"""

from __future__ import annotations

import csv
import math
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

CONTRACT_PATH = Path("ai_agent/experiment.yaml")
DRAFT_NAME = "draft.md"                 # working name before the id (sha) exists

# Project-specific: the MLflow store the training script writes to. Must match
# src/tracking.py / the `mlflow ui --backend-store-uri` path. The one line to
# change if this driver is reused on another project.
MLFLOW_TRACKING_URI = "mlruns"


class ScopeViolation(Exception): ...
class EvalFailed(Exception): ...


# -----------------------------------------------------------------------------
# Contract
# -----------------------------------------------------------------------------
def load_contract(path: Path = CONTRACT_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# -----------------------------------------------------------------------------
# Git plumbing
# -----------------------------------------------------------------------------
def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=check)


def _git_head() -> str:
    return _git("rev-parse", "HEAD").stdout.strip()


def _git_changed_files() -> list[str]:
    """Every path git considers dirty vs the index/HEAD, INCLUDING untracked
    files (so a brand-new frozen file is caught too). Gitignored files (e.g. the
    heavy .pth artifacts) are excluded by default -- exactly what we want."""
    out = _git("status", "--porcelain").stdout
    files = []
    for ln in out.splitlines():
        if not ln.strip():
            continue
        path = ln[3:]                        # strip the two status chars + space
        if " -> " in path:                   # rename: keep the destination
            path = path.split(" -> ", 1)[1]
        files.append(path.strip().strip('"'))
    return files


def _ensure_branch(branch: str) -> None:
    """Switch to the experiment branch (carrying the agent's uncommitted edits),
    creating it from the current HEAD if it does not exist yet."""
    r = _git("checkout", "-q", branch, check=False)
    if r.returncode != 0:
        _git("checkout", "-q", "-b", branch)


def git_commit(msg: str) -> str:
    """Stage everything, commit, return the SHORT sha (used as the trial id)."""
    _git("add", "-A")
    _git("commit", "-q", "-m", msg)
    return _git("rev-parse", "--short", "HEAD").stdout.strip()


def _revert_mutable_to(sha: str, mutable: list[str]) -> None:
    """Restore the mutable files to their pre-trial content (the champion's).
    NOTE: this must target the PRE-trial sha, not HEAD -- after commit 1 the bad
    code is committed, so `git checkout -- <path>` would restore the bad code."""
    for path in mutable:
        _git("checkout", sha, "--", path, check=False)


# -----------------------------------------------------------------------------
# 1. Scope check -- ALLOWLIST (deny by default)
# -----------------------------------------------------------------------------
def assert_only_mutable_changed(mutable: list[str], experiments_dir: Path) -> None:
    """Only the mutable allowlist and the trial record dir may differ from HEAD.
    Everything else -- including experiment.yaml, driver.py and program.md, which
    live under ai_agent/ but OUTSIDE experiments_dir -- is frozen. This is what
    makes the judge itself immutable."""
    exempt = str(experiments_dir).rstrip("/") + "/"
    allowed = set(mutable)
    illegal = [f for f in _git_changed_files()
               if f not in allowed and not f.startswith(exempt)]
    if illegal:
        raise ScopeViolation(
            f"Modified frozen files: {illegal}. Only {mutable} and {exempt}* "
            f"may change. Revert the stray change(s) and rerun."
        )


# -----------------------------------------------------------------------------
# Record frontmatter
# -----------------------------------------------------------------------------
def read_frontmatter(md_path: Path) -> dict:
    m = re.match(r"^---\n(.*?)\n---\n", md_path.read_text(), re.DOTALL)
    if not m:
        raise ValueError(f"No frontmatter found in {md_path}")
    return yaml.safe_load(m.group(1)) or {}


# -----------------------------------------------------------------------------
# 3. Repeat axis + eval launch
# -----------------------------------------------------------------------------
def _repeat_axis(cfg: dict) -> str | None:
    ro = cfg["training"].get("repeat_over")
    return next(iter(ro)) if ro else None


def _repeat_values(cfg: dict) -> list[dict]:
    """Per-run overrides. [{}] means a single run (N=1)."""
    ro = cfg["training"].get("repeat_over")
    if not ro:
        return [{}]
    (axis, values), = ro.items()                 # e.g. ("latent_dimensions", [8,60,240])
    return [{axis: v} for v in values]


def run_eval(cfg: dict, trial_id: str, overrides: list[dict], log_path: Path) -> None:
    """Launch the N trainings. All-or-nothing: the first non-zero exit aborts."""
    base = cfg["training"]["command"].split()
    with open(log_path, "w") as logf:
        for ov in overrides:
            cmd = list(base)
            cmd += ["--trial-id", trial_id]      # HOOK: run_autoencoder.py must set MLflow tag trial_id=<id>
            for k, v in ov.items():
                cmd += ["--set", f"{k}={v}"]     # HOOK: run_autoencoder.py must override cfg[k]=v in memory
            logf.write(f"\n=== {' '.join(cmd)} ===\n"); logf.flush()
            proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
            if proc.returncode != 0:
                raise EvalFailed(
                    f"Training failed (exit {proc.returncode}) for override={ov}. "
                    f"See {log_path}."
                )


# -----------------------------------------------------------------------------
# 4. Read the trial's runs BY TAG, then aggregate
# -----------------------------------------------------------------------------
def _coerce(s):
    try:
        return int(s)
    except (TypeError, ValueError):
        try:
            return float(s)
        except (TypeError, ValueError):
            return s


def read_trial_runs(trial_id: str, per_run_metric: str,
                    also_log: list[str], axis: str | None) -> list[dict]:
    """Exactly the runs tagged with this trial_id (never by recency).
    Returns one dict per run: {run_id, axis_value, <per_run_metric>, <also_log...>},
    sorted by the repeat-axis value when it is numeric."""
    import mlflow
    import pandas as pd

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    df = mlflow.search_runs(search_all_experiments=True,
                            filter_string=f"tags.trial_id = '{trial_id}'")

    records = []
    for _, row in df.iterrows():
        rec = {"run_id": row["run_id"], "axis_value": None}
        if axis:
            pcol = f"params.{axis}"
            if pcol in df.columns and pd.notna(row[pcol]):
                rec["axis_value"] = _coerce(row[pcol])
        for name in [per_run_metric, *also_log]:
            mcol = f"metrics.{name}"
            rec[name] = float(row[mcol]) if mcol in df.columns and pd.notna(row[mcol]) else float("nan")
        records.append(rec)

    def _key(r):
        try:
            return (0, float(r["axis_value"]))
        except (TypeError, ValueError):
            return (1, str(r["run_id"]))
    records.sort(key=_key)
    return records


def _aggregate(values: list[float], how: str) -> float:
    if how == "identity":
        if len(values) != 1:
            raise ValueError("aggregation 'identity' expects exactly 1 run")
        return values[0]
    if how == "mean":
        return sum(values) / len(values)
    raise ValueError(f"Unknown aggregation '{how}'. Use 'identity' or 'mean'.")


# -----------------------------------------------------------------------------
# 5. Verdict -- deterministic, NO LLM
# -----------------------------------------------------------------------------
def _beats(candidate: float, champion: float, direction: str) -> bool:
    return candidate > champion if direction == "maximize" else candidate < champion


def read_champion_metric(ledger: Path, direction: str) -> float | None:
    """Best kept aggregate so far. Filters on verdict, NOT on 'kept-ness':
    CANDIDATE rows are kept on disk but are NOT champions."""
    if not ledger.exists():
        return None
    best = None
    with open(ledger) as f:
        for row in csv.DictReader(f):
            if row.get("verdict") in ("BASELINE", "CHAMPION"):
                try:
                    v = float(row["metric_value"])
                except (TypeError, ValueError):
                    continue
                if best is None:
                    best = v
                else:
                    best = max(best, v) if direction == "maximize" else min(best, v)
    return best


def decide_verdict(aggregate: float, per_run_values: list[float],
                   champion: float | None, cfg: dict) -> str:
    direction = cfg["eval"]["direction"]
    if champion is None:
        return "BASELINE"
    if _beats(aggregate, champion, direction):
        return "CHAMPION"
    cand = cfg["decision"].get("candidate")
    if cand:
        stat = cand.get("statistic", "max")
        margin = float(cand.get("margin", 0.0))
        best_run = max(per_run_values) if stat == "max" else min(per_run_values)
        if direction == "maximize" and best_run > champion + margin:
            return "CANDIDATE"
        if direction == "minimize" and best_run < champion - margin:
            return "CANDIDATE"
    return "FAILURE"


def trial_count(ledger: Path) -> int:
    """Data rows in the ledger = trials in the current campaign (all verdicts,
    failures included). Resets to 0 only when the CSV is archived+emptied."""
    if not ledger.exists():
        return 0
    with open(ledger) as f:
        return sum(1 for _ in csv.DictReader(f))


# -----------------------------------------------------------------------------
# 6. Record + ledger
# -----------------------------------------------------------------------------
LEDGER_HEADER = ["timestamp", "id", "parent", "model_name", "modification_description",
                 "metric_name", "metric_value", "metric_delta", "status", "verdict"]


def append_ledger_row(ledger: Path, row: dict) -> None:
    """Append one row. The driver owns the header: written on the first trial,
    skipped afterwards. Do NOT create this file by hand."""
    exists = ledger.exists()
    with open(ledger, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LEDGER_HEADER)
        if not exists:
            w.writeheader()
        w.writerow(row)


def _replace_section(body: str, heading: str, content: str) -> str:
    """Replace the text under '## heading' (up to the next '## ' or EOF),
    keeping the heading line. Appends the section if absent."""
    pat = re.compile(rf"(^## {re.escape(heading)}\n)(.*?)(?=^## |\Z)",
                     re.DOTALL | re.MULTILINE)
    if pat.search(body):
        return pat.sub(lambda m: m.group(1) + content.rstrip() + "\n\n", body)
    return body.rstrip() + f"\n\n## {heading}\n{content.rstrip()}\n"


def _update_title(body: str, trial_id: str, model_name: str, label: str) -> str:
    return re.sub(r"^# Trial .*$",
                  f"# Trial {trial_id} — {model_name or '?'} — {label}",
                  body, count=1, flags=re.MULTILINE)


def write_record(md_path: Path, trial_id: str, status: str, verdict: str | None,
                 aggregate: float | None, runs: list[dict] | None,
                 cfg: dict, created_at: str, delta: float | None,
                 error: str | None = None) -> None:
    """Fill the frontmatter + ## Results. Numbers by the driver; the agent adds
    ## Training Dynamics and ## Conclusion afterwards."""
    text = md_path.read_text()
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.DOTALL)
    if not m:
        raise ValueError(f"No frontmatter found in {md_path}")
    fm = yaml.safe_load(m.group(1)) or {}
    body = m.group(2)

    fm["id"] = trial_id
    fm["status"] = status
    fm["verdict"] = verdict
    fm["created_at"] = created_at
    if aggregate is not None:
        fm["metric"] = {"primary": {
            "name": cfg["eval"]["primary_metric_name"],
            "value": round(float(aggregate), 6),
            "direction": cfg["eval"]["direction"],
        }}

    if status == "failed":
        results = (f"- **Trial failed mechanically** — {error}\n"
                   f"- No comparable metric produced; mutable files reverted to the pre-trial state.")
        label = "FAILED"
    else:
        per = cfg["eval"]["per_run_metric"]
        also = cfg["eval"].get("also_log", []) or []
        per_parts = []
        for r in runs:
            tag = r["axis_value"] if r["axis_value"] is not None else r["run_id"][:8]
            per_parts.append(f"{tag}: {r[per]:.6f}")
        lines = [
            f"- **{per} per run:** " + " | ".join(per_parts),
            f"- **{cfg['eval']['primary_metric_name']}:** {aggregate:.6f}",
            f"- **delta_vs_champion** (display only): {delta:+.6f}",
        ]
        for name in also:
            vals = [r[name] for r in runs]
            lines.append(f"- **{name}** (mean, non-decisional): {sum(vals) / len(vals):.6f}")
        lines.append("- **MLflow Run IDs:** " + " ".join(r["run_id"] for r in runs))
        results = "\n".join(lines)
        label = verdict

    body = _replace_section(body, "Results", results)
    body = _update_title(body, trial_id, fm.get("model_name", ""), label)
    new_fm = yaml.dump(fm, sort_keys=False, default_flow_style=False, allow_unicode=True)
    md_path.write_text(f"---\n{new_fm}---\n{body}")


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------
def run_trial() -> tuple[str, str | None]:
    cfg = load_contract()
    exp_dir = Path(cfg["logging"]["experiments_dir"])
    ledger = Path(cfg["logging"]["ledger"])
    template = cfg["logging"]["template"]
    branch = cfg["decision"]["branch"]
    mutable = cfg["mutable"]
    direction = cfg["eval"]["direction"]

    draft = exp_dir / DRAFT_NAME
    if not draft.exists():
        raise FileNotFoundError(
            f"{draft} not found. Copy {template} to {draft} and fill "
            "Hypothesis / Implementation / model_name / parent before running."
        )

    # 0. campaign cap -- refuse before any work (no commit, no training)
    max_trials = cfg["decision"].get("max_trials")
    if max_trials is not None and trial_count(ledger) >= max_trials:
        print(f"Reached max_trials={max_trials} for this campaign. Archive "
              f"{ledger} + reports and start a new campaign to continue.")
        return ("skipped", None)

    _ensure_branch(branch)

    # 1. scope check (pre-flight: nothing is committed if this fails)
    assert_only_mutable_changed(mutable, exp_dir)

    # remember the pre-trial (champion) code, to revert to on failure
    parent_sha = _git_head()

    # 2. commit 1 (input) -> id = short sha ; rename draft -> <id>.md
    trial_id = git_commit("[trial] lock input")
    md_path = exp_dir / f"{trial_id}.md"
    if md_path.exists():
        raise ValueError(f"Record {md_path} already exists (sha collision?). Aborting.")
    draft.rename(md_path)

    fm = read_frontmatter(md_path)
    model_name = fm.get("model_name", "")
    summary = fm.get("summary", "")
    parent = fm.get("parent") or ""
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    log_path = exp_dir / f"{trial_id}.console.log"

    try:
        # 3. train (N runs, all-or-nothing)
        overrides = _repeat_values(cfg)
        run_eval(cfg, trial_id, overrides, log_path)

        # 4. read the N runs by tag + aggregate
        per = cfg["eval"]["per_run_metric"]
        also = cfg["eval"].get("also_log", []) or []
        runs = read_trial_runs(trial_id, per, also, _repeat_axis(cfg))
        if len(runs) != len(overrides):
            raise EvalFailed(f"Expected {len(overrides)} runs tagged {trial_id}, found {len(runs)}.")
        values = [r[per] for r in runs]
        if cfg.get("safety", {}).get("nan_abort") and any(math.isnan(v) for v in values):
            raise EvalFailed("NaN in a per-run metric.")
        aggregate = _aggregate(values, cfg["eval"]["aggregation"])

        # 5. verdict (deterministic)
        champion = read_champion_metric(ledger, direction)
        delta = (aggregate - champion) if champion is not None else 0.0
        verdict = decide_verdict(aggregate, values, champion, cfg)
        if verdict == "FAILURE":
            _revert_mutable_to(parent_sha, mutable)

        # 6. record + ledger
        write_record(md_path, trial_id, "completed", verdict, aggregate, runs,
                     cfg, created_at, delta)
        append_ledger_row(ledger, {
            "timestamp": created_at, "id": trial_id, "parent": parent,
            "model_name": model_name, "modification_description": summary,
            "metric_name": cfg["eval"]["primary_metric_name"],
            "metric_value": f"{aggregate:.6f}", "metric_delta": f"{delta:+.6f}",
            "status": "completed", "verdict": verdict,
        })

        # 7. commit 2 (output)
        git_commit(f"[trial] {trial_id} {verdict} ({aggregate:.4f})")
        print(f"Trial {trial_id}: {verdict}  "
              f"{cfg['eval']['primary_metric_name']}={aggregate:.4f}  delta={delta:+.4f}")
        return ("completed", verdict)

    except Exception as e:
        # mechanical failure: revert code, still record + commit to leave a trace
        _revert_mutable_to(parent_sha, mutable)
        write_record(md_path, trial_id, "failed", None, None, None,
                     cfg, created_at, None, error=f"{type(e).__name__}: {e}")
        append_ledger_row(ledger, {
            "timestamp": created_at, "id": trial_id, "parent": parent,
            "model_name": model_name, "modification_description": summary,
            "metric_name": cfg["eval"]["primary_metric_name"],
            "metric_value": "", "metric_delta": "",
            "status": "failed", "verdict": "",
        })
        git_commit(f"[trial] {trial_id} FAILED ({type(e).__name__})")
        print(f"Trial {trial_id}: FAILED — {type(e).__name__}: {e}", file=sys.stderr)
        return ("failed", None)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd != "run":
        print(f"Unknown command: {cmd}. Use: python ai_agent/driver.py run", file=sys.stderr)
        sys.exit(1)
    try:
        status, _ = run_trial()
    except (ScopeViolation, FileNotFoundError, ValueError) as e:
        # pre-flight problems: nothing was committed; fix and rerun
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(2)
    sys.exit(1 if status == "failed" else 0)