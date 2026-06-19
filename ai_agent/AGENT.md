# AGENT.md

## Project Vision
History: the user has developped models (PCA, autoencoders, etc.) for dimensionality reduction on IRM heart images for 150 patients. 
The autoencoders have various architectures and hyperparameters, but despite the user's effort to optimize them (through Optuna 
for hyperparameters for example), the autoencoders still perform poorly compared to simpler models like PCA.

We are working on a Renku session, where a AI Agent pi (you) is built from the race-sdsc-llm-poc folder. 
The goal is to use this Agent to optimize models (autoencoders whose architecture and hyperparameters could be improved) in the SDSC-IRM folder 
through iterations.

## Operational Procedures
The SDSC-IRM code has been cloned from a Github repo, and will be branched (TO DO !).
The user will then give you directions on the levers you will be allowed to use to optimize the performances of the autoencoders. Examples: 
architecture, some parameters (dropout rate, patience, learning rate, etc.), etc.
Then the AI Agent (you) will modify the code in SDSC-IRM (just the yaml file for parameters, or actual .py files for deeper changes like architectures)
and run the autoencoder training with run_autoencoder.py
You will then use a metric (TO BE DEFINED!) to see if the change improved or worsened the performance of the model.
If the performance has improved, commit the change in the Git branch. If not, revert the change, and try something ELSE (keep in mind what you tried 
not to try the same modification over and over)

## Technical Environment
[Insert paths, environment activation commands, and dependency information here]

## Rules
- MOST IMPORTANT RULE: do not modify files unless explicitly asked to!!!
- Do not commit unless part of a specified trial/evaluation/commit procedure to improve AE models.
- Always verify changes with tests

## Communication Style
- Be concise and direct.
- Please don't have a validation bias, you can challenge the user's idea if you think they are suboptimal.

## To-do list 
GENERAL
- Implement Git branching
- Train an AE and test commit on branch 
AUTOENCODERS ARCHITECTURE
- Test change of AE architecture 
- Choose evaluation metric and 
- Test training with architecture changed, and evaluation (+ commit or not, depending on the result)
- Launch AE architecture optimization on 20 trials 
KARPATHY AUTORESEARCH
- Branch Karpathy Autoresearch as a 3rd code repo 
- Automatize trial/evaluation/commit loop 
- Make the process "independent" from code to optimize (SDSC-IRM here)