<!--
This file is written by the HUMAN and read by the AGENT. It holds INTENT, not
hard rules. All machine-enforced rules (mutable files, the metric, the budget,
the keep/revert verdict) live in experiment.yaml and are enforced by driver.py.
Keep the two separate: prose here explains WHY and WHAT to explore; the YAML
decides and the driver applies. Rewrite this file at the start of each campaign.
-->

# program.md — Research Session Intention

## Objective
<!-- EDIT PER CAMPAIGN. Example:
Find the autoencoder ARCHITECTURE that maximizes avg_validation_R2_mean (mean
validation R² over latent dims 8 / 60 / 240), hyperparameters held fixed. This is
an optimization goal, not a controlled comparison — we want the best model. -->

## What to explore this session
<!-- EDIT PER CAMPAIGN. Examples:
- Refine the current champion, or explore a new architectural family — see the
  exploration/exploitation balance you want.
- Combining ideas is encouraged: if you merge two prior directions (e.g. attention
  + dilated convs), set `parent` to the trial whose CODE you branched from, and
  describe the fusion explicitly in ## Hypothesis.
- Motivate every change: each trial costs real compute, so propose deliberate,
  mechanistically-justified changes — not random search.
- Architectural constraint for this project: NO skip connections (no U-Net) — all
  information must pass through the bottleneck. -->

## Hard boundaries (defined in experiment.yaml, enforced by the driver — see SETUP.md)
- Modify only files in the `mutable` allowlist. Everything else — including
  experiment.yaml, driver.py and this file — is frozen; the driver rejects any
  trial that touches it, before committing or training.
- The judge (metric + verdict rule) is fixed for the whole campaign. Wanting a
  different metric means a new campaign, not an edit mid-run.

## How to run a trial
1. Read the current champion: the best `keep` row in `trial_log.csv` — i.e. the
   highest-metric row whose `verdict` is CHAMPION or BASELINE — and open its
   `experiments/<id>.md` for context. (The driver reads the champion the same way.)
2. Create a new draft from the template (a real file copy — do not reconstruct the
   structure by hand):
   ```bash
   cp ai_agent/experiments/TEMPLATE.md ai_agent/experiments/draft.md
   ```
   Then fill `model_name`, `summary`, `parent` in the frontmatter and the
   ## Hypothesis + ## Implementation prose BEFORE running. Leave the
   driver-written fields at null.
3. Edit a mutable file (see Hard boundaries).
4. Run the driver, then wait — do not edit anything while it trains:
   ```bash
   python ai_agent/driver.py run
   ```
   No `parent` argument: lineage is the `parent` field you filled in step 2. The
   driver commits the input, renames draft.md to `<id>.md`, runs the N trainings,
   decides the verdict, and writes ## Results.
5. When it returns, read the verdict and write ## Training Dynamics and
   ## Conclusion in the `<id>.md`. Then start the next trial.

## How to run a campaign
A campaign = repeating the trial cycle autonomously until the budget is spent. The
human starts the loop (e.g. from the CLI: "run trials until the driver refuses, don't
ask me between trials"). You then iterate WITHOUT waiting for confirmation:

  repeat steps 1–5 of "How to run a trial"
  until `python ai_agent/driver.py run` prints "Reached max_trials=N".

The stop condition is that driver message — not a count you keep in your head. When it
appears, the campaign is COMPLETE: stop and write a short summary (the champion, and
what was learned across the trials).

Between trials there is nothing to reset or clean. The driver has already handled a
FAILURE end-to-end: reverted its code, purged its MLflow runs, and committed the
result (its `<id>.md` + CSV row + `<id>.console.log` remain as the trace). Just start
the next trial by recreating draft.md (step 2). Do NOT run any destructive git or
filesystem cleanup yourself.

Stop early — and say so clearly — only if:
- you keep proposing variations of an idea that repeatedly fails (no loops), or
- a run fails twice in a row for the same reason.