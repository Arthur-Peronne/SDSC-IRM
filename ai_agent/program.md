# program.md — Research Session Intention

<!--
This file is written by the HUMAN and read by the AGENT. It holds INTENT, not
hard rules. All machine-enforced rules (which files are mutable, the metric,
the budget, keep/revert) live in experiment.yaml and are enforced by driver.py.
Keep the two separate: prose here explains WHY and WHAT to explore; the YAML
decides and the driver applies. Rewrite this file each session.
-->

## Objective : GOAL TO MODIFY AP
<!-- Find the single best autoencoder — architecture **and** hyperparameters together
— that maximizes `avg_validation_R2_mean` (mean validation R² over latent dims
8 / 60 / 240). This is an OPTIMIZATION goal, not a controlled comparison: you may
change architecture and hyperparameters at the same time. We want the best model,
not the isolated effect of any one factor. -->

## What to explore this session : STRATEGY TO MODIFY AP
<!-- Edit per session. Examples: -->
<!-- - Refine the current champion rather than exploring from scratch. -->
<!-- - Combining ideas is encouraged. If you merge two prior directions into one
  architecture (e.g. attention + dilated convs), set `parent` to the trial you
  branched the CODE from, and describe the fusion explicitly in ## Hypothesis. -->
<!-- - Motivate every change: each trial costs ~30 min, so propose deliberate,
  mechanistically-justified changes — not random search. -->

## Hard boundaries (defined in experiment.yaml, enforced by the driver)
- Modify only files in the mutable allowlist. Everything else is frozen by
default — the driver rejects any trial that touches it. See SETUP.md.

## How to run a trial
1. Read the current champion (top-scoring `keep` row in trial_log.csv) and the
   relevant <hash>.md.
2. Copy `experiments/TEMPLATE.md` to `experiments/draft.md` — always start from
   the template, never reinvent the record structure. Fill ## Hypothesis,
   ## Implementation, and `model_name` in the frontmatter BEFORE running.
3. Edit a mutable file (see Hard boundaries section).
4. Run `python ai_agent/driver.py run <parent_id>`. Then wait — do not edit
   anything while the driver trains. The driver commits the input, renames
   draft.md to <hash>.md, runs the N trainings, and writes ## Results.
5. When it returns, write ## Training Dynamics and ## Conclusion. Next trial.

## When to stop
- Stop and flag if you are repeating variations of a failed idea (no loops).
- Stop if a run crashes twice in a row for the same reason.