# AGENT.md

## Project Vision
History: the user has developped models (PCA, autoencoders, etc.) for dimensionality reduction on IRM heart images for 150 patients. 
The autoencoders have various architectures and hyperparameters, but despite the user's effort to optimize them (through Optuna 
for hyperparameters for example), the autoencoders still perform poorly compared to simpler models like PCA.

We are working on a Renku session, where a AI Agent pi (you) is built from the race-sdsc-llm-poc folder. 
The goal is to use this Agent to optimize models (autoencoders whose architecture and hyperparameters could be improved) in the SDSC-IRM folder 
through iterations.

## Operational Procedures
The SDSC-IRM code has been cloned from a Github repo, and is branched (branch name: ae-agent-opti)

The user will then give you directions on the levers you will be allowed to use to optimize the performances of the autoencoders. Examples: 
architecture, some parameters (dropout rate, patience, learning rate, etc.), etc. (see Rules and goals for AE optimization)

**NOTE ON HYPERPARAMETERS:** For the current 20-trial architecture optimization loop, hyperparameters are FROZEN. Do not change them. Hyperparameter optimization will be a separate, subsequent experiment.

Then the AI Agent (you) will modify the code in SDSC-IRM (just the yaml file for parameters, or actual .py files for deeper changes like architectures).
And then effectively train the model. 
**When running training, always use unbuffered output and tee to allow real-time monitoring:**
`python -u scripts/run_autoencoder.py | tee training_<experiment_tag>.log`

Save your trial results in trial_log.csv.
If the performance has improved, commit the change in the Git branch. If not, revert the change, and try something ELSE (trial_log.csv helps you to 
keep in mind what you tried not to try the same modification over and over)

## Rules and goals for AE optimization

### Optimization Goals
- **Primary Metric:** Use the **Reconstruction Loss (MSE) on the validation set (`val_mse`)** to decide if a change is successful. 
- **Objective:** Minimize `val_mse` to improve image reconstruction quality (reducing blurriness) and to avoid instabilities in VAEs (like beta collapsing to zero).

### Core Architectural Rules
- **No U-Net (No skip connections):** The architecture must maintain an independent latent space. Information must pass through the bottleneck alone. This is crucial to allow future constraints to be applied directly to the latent space to study heart tissue deformations.

## Technical Environment
[Insert paths, environment activation commands, and dependency information here]

## Rules
- MOST IMPORTANT RULE: do not modify files unless explicitly asked to!!!
- Do not commit unless part of a specified trial/evaluation/commit procedure to improve AE models.
- Always verify changes with tests

### Loop-Breaking Protocol (To avoid infinite retry loops)
- **Two-Strike Rule:** If an `edit` fails twice, do not attempt a third time using the same logic.
- **Mandatory Re-Read:** If an `edit` fails, immediately `read` the file again to ensure current state is captured.
- **Granularity Shift:** If large edits fail, switch to "Micro-Edits" (replacing 1–3 lines instead of large blocks).
- **Strategy Pivot:** If micro-edits fail, use `bash` (e.g., `cat -A`) to inspect whitespace/hidden characters, or suggest a full `write`. If still stuck, stop and ask the user for guidance.

## Communication Style
- Be concise and direct.
- Please don't have a validation bias, you can challenge the user's idea if you think they are suboptimal.

## To-do list 
GENERAL
- Implement Git branching : DONE
- Train an AE and test commit on branch : DONE
AUTOENCODERS ARCHITECTURE
- Choose evaluation metric (DONE: val_mse)
- Test change of AE architecture (DONE: AE3dAttention)
- Test training with architecture changed, and evaluation (+ commit or not, depending on the result) (DONE: Trial 1 committed)
- Launch AE architecture optimization on 20 trials
KARPATHY AUTORESEARCH
- Branch Karpathy Autoresearch as a 3rd code repo 
- Code "model independent": refactoring to use kwargs instead of hard coded parameters.
- Automatize trial/evaluation/commit loop 
- Make the process "independent" from code to optimize (SDSC-IRM here)