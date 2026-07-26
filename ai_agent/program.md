<!--
This file is written by the HUMAN and read by the AGENT. It holds INTENT, not
hard rules. All machine-enforced rules (mutable files, the metric, the budget,
the keep/revert verdict) live in experiment.yaml and are enforced by driver.py.
Keep the two separate: prose here explains WHY and WHAT to explore; the YAML
decides and the driver applies. Rewrite this file at the start of each campaign.
-->

# program.md — Research Session Intention

## Objective
The goal of this campaign is to optimize the hyperparameters of the autoencoder "AE3dAsymResSeparableV2" at latent_dimensions = 20, by running trials in order
to maximize the primary metric.

## What to explore this session
- Modify only the hyperparamers autoencoder.yaml (you can't change the model architecture or other parameters)
- Only 5 hyperparameters opened to change, to optimize: lr, weight_decay, dropout_rate, noise_std, patience
- hyper_automatic_values is already set to false for this campaign (so the YAML values are the ones used). Do NOT change it, and do NOT touch any field of autoencoder.yaml other than the ones opened to change.
- Propose deliberate, mechanistically-justified changes — not random search.

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

## Known noise floor (measured empirically, not a strategy hint)
Prior campaigns on this architecture/pipeline (batch_size=1, seed fixed) showed
substantial run-to-run variance even with identical config and seed — roughly
0.03-0.04 standard deviation on avg_validation_R2_mean (and unknown for classification 
accuracy, possibly more), likely from non-deterministic
GPU ops. A single-run delta smaller than this is not reliably distinguishable from
noise.

## Budget discipline
You have exactly `max_trials` trials for this campaign (see `decision.max_trials` in
experiment.yaml), no more. Move decisively from the first trial — don't spend early
trials re-deriving things you can already reason about from config/code or from the
noise-floor note above. Pace your exploration so that by the last trial you have
converged on your best candidate, not still mid-sweep.