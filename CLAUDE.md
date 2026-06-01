# CLAUDE.md — SDSC Arthur Péronne

## Projet
Analyse de données IRM cardiaques 3D. Le pipeline va du preprocessing des images (recalage, rééchantillonnage) jusqu'à la comparaison de méthodes de réduction de dimensionnalité (PCA spatiale, autoencodeurs 3D) et régression sur métadonnées patients.

Les données médicales brutes sont sur S3 (AWS) et accessibles uniquement via Renku. Le développement se fait en local, les tests sur données réelles se font sur Renku après commit/push.

## Commandes utiles

```bash
# MLflow UI en local (depuis la racine du projet, venv activé)
mlflow ui --backend-store-uri mlruns/
# → http://127.0.0.1:5000

# MLflow UI sur Renku : ne fonctionne pas (VSCode Server, pas JupyterLab)
# Fallback : search_runs() dans un terminal Python
```

## Règles de travail
- **Langue** : discussions en français, code en anglais
- **Priorité absolue** : reproductibilité des résultats passés — ne jamais casser ce qui fonctionne
- **Refactoring** : conserver les anciens fichiers dans `archive_scripts/` pendant la réécriture, supprimer seulement quand le nouveau code est validé
- **Nouveaux fichiers** : propres dès le départ (imports corrects, pathlib, docstrings)
- **Pas de venv sur Renku** (environnement Docker) — en local : `.venv/`

## Conventions de code
- Imports : `from src.config import ...` (jamais de `from paths import *`)
- Chemins : `pathlib.Path` avec `/` (jamais de concaténation de strings)
- Configuration centralisée dans `src/config.py` via `python-dotenv`
- Package installé en mode éditable : `pip install -e .`
- Tracking des expériences : MLflow via `src/tracking.py` (jamais d'appels MLflow directs dans les scripts)

## Arborescence
```
SDSC-arthur-project-1/
├── archive_scripts/       ← anciens scripts conservés pendant le refactoring, à supprimer une fois validé
├── experiments/           ← scripts de test/exploration
├── scripts/               ← scripts exécutables (préfixe run_)
├── src/
│   ├── config.py          ← chemins et variables d'environnement
│   ├── tracking.py        ← fonctions utilitaires MLflow (init, log params/metrics/model)
│   ├── data/
│   │   ├── loader.py      ← chargement unifié nii + splits train/val/test (partagé PCA et AE)
│   │   └── ...            ← preprocessing (geometry, registration, resampling...)
│   ├── models/            ← PCA, autoencodeurs, régression
│   ├── training/          ← entraînement AE, Optuna
│   └── visualization/     ← plots
├── mlruns/                ← MLflow : modèles entraînés + métriques (ne pas commiter)
└── processed_images/      ← fichiers nii resamplés/croppés/registrés (ne pas commiter)
```

## Gestion des données
- **S3 (AWS)** : données raw uniquement, accessibles via Renku
- **processed_images/** : fichiers nii générés une seule fois (coûteux), rechargés manuellement entre sessions Renku
- **mlruns/** : tous les modèles entraînés + métriques via MLflow, rechargés manuellement entre sessions Renku
- Ni `mlruns/` ni `processed_images/` ne vont dans git

## Fonctionnalités scientifiques clés
- **Visualisation MRI** : plot nii files (image, masque, superposition), multi-timeframes
- **Pipeline preprocessing** : resampling → cropping → registration (SimpleITK) + checks DICE
- **PCA temporelle** : par patient, sur fichiers 4D
- **PCA spatiale** : across patients, supports ED / ES / ED+ES, plusieurs types d'images
- **Autoencodeurs 3D** : plusieurs architectures (`AE3dFCDeep`, `AE3dLinear`...), early stopping, Optuna pour hyperparamètres
- **Régression** : logistique et linéaire sur coordonnées PCA + AE + métadonnées patients
- **Comparaisons AE vs PCA** : mêmes splits train/val/test, lecture directe depuis MLflow

## Questions ouvertes à trancher pendant le refactoring
- Mean subtraction dans PCA : sklearn `PCA` centre les données par défaut — vérifier que ce n'est pas fait en double dans le code actuel
- Eigenvectors spatiaux → nii : quel affine/header utiliser ? (proposition : patient de référence)
- Stats early stopping AE : dans `run_autoencoder.py` ou `run_comparison_aepca.py` ?
- Nommage des expériences MLflow : par pipeline (`pca_spatial`, `autoencoder`) ou par expérience scientifique ?

## État du refactoring

### Terminé ✅
- Arborescence `src/` restructurée
- Tous les imports migrés vers `from src.config import ...`
- Chemins migrés vers `pathlib.Path`
- `config.py` avec python-dotenv
- `pyproject.toml` + installation en mode éditable
- Sécurité git (repo privé, .gitignore, clés AWS régénérées)

---

### Pan 1 — Infrastructure ✅

**Étape 1 : Mettre en place MLflow** ✅
- Ajouter MLflow aux dépendances (`requirements_locked.txt`)
- Créer `src/tracking.py` avec fonctions utilitaires (init experiment, log params/metrics/model)
- Définir convention de nommage des expériences et runs
- Tester sur un run simple

**Étape 1b : Gestion des paramètres via YAML** ✅
- Créer `configs/` à la racine avec un fichier par pipeline :
  `pca_spatial.yaml`, `pca_temporal.yaml`, `registration.yaml`,
  `autoencoder.yaml`, `regression.yaml`, `comparison.yaml`
- Chaque fichier contient tous les paramètres d'entrée du pipeline (données, options, hyperparamètres, experiment_tag)
- Chaque `run_xxx.py` lit son YAML au démarrage via `yaml.safe_load()`
- MLflow archive le YAML entier comme artefact à chaque run → traçabilité complète
- PyYAML déjà présent dans les dépendances (`PyYAML==6.0.3`)
- Workflow : modifier le YAML → lancer le script → MLflow archive automatiquement

**Étape 2 : Réorganiser les dossiers de données** ✅
- Remplacer `tempodata/` par `mlruns/` et `processed_images/`
- Mettre à jour `config.py` avec les nouveaux chemins
- Mettre à jour `.gitignore`
- Supprimer `requirements.txt` et `requirements-OLD.txt`, garder uniquement `requirements_locked.txt`
- Vérifier/supprimer `experiments/test.py`

**Étape 3 : Module de chargement des données unifié** ✅
- Créer `src/data/loader.py` : chargement nii + splits train/val/test
- Inputs : process level, image type (full/ROIonly/mask/binarymask), ED/ES, splitname
- Output : X array/tensor + X_train/X_val/X_test
- Remplace le code dupliqué entre PCA spatiale et AE

---

### Pan 2 — Fonctionnel

**Étape 4 : Visualisation MRI** — `run_visualize.py` (obsolète → réécrire)
- Trouver nii depuis paramètres (type de données, numéro patient, type de fichier)
- Plot image / masque / superposition, 1 ou plusieurs timeframes
- Output : PDF dans `results/`

**Étape 5 : Image registration** — `run_registration.py` + `run_registration_pipelinechecks.py`
- Resampling → Cropping → Registration
- Checks : shape/spacing, DICE avant/après registration
- Outputs dans `processed_images/registered_frames/` uniquement (`registered_framesBIS` obsolète → supprimer toutes les références)

**Étape 6 : PCA temporelle** — `run_pca_temporal.py`
- Extraction voxels depuis 4D nii par patient
- Calcul PCA + plots (variance expliquée, eigenbase 2D, eigenvectors, reconstruction)
- Outputs modèle + métriques dans MLflow

**Étape 7 : PCA spatiale** — `run_pca_spatial.py`
- Utilise `loader.py` unifié
- Migrer `pca_spatial.py` de `TEMPODATA_FOLDER` vers `PROCESSED_IMAGES_FOLDER` (migration tempodata → processed_images)
- Calcul PCA + métriques (R², etc.) sur train/val/test
- Plots (variance expliquée, eigenbase 2D, eigenvectors, reconstruction)
- Outputs dans MLflow

**Étape 8 : Autoencoder** — `run_autoencoder.py` + `run_ae_hyperparam.py`
- Utilise `loader.py` unifié
- Entraînement AE (plusieurs architectures, early stopping)
- Optuna pour optimisation hyperparamètres
- Métriques (R², etc.) sur train/val/test
- Plots (loss history, reconstructions)
- Outputs dans MLflow

**Étape 9 : Régression** — `run_regression.py`
- From PCA results (refactoring de l'existant)
- From AE results (à écrire)
- Métriques (accuracy, R², confusion matrix) + plots
- Outputs dans MLflow

**Étape 10 : Comparaisons AE vs PCA** — `run_comparison_aepca.py`
- Lecture directe depuis MLflow via `search_runs()` (remplace le parsing de `.txt`)
- Comparaisons AE-AE (architectures, ED vs ED+ES, baseline vs Optuna)
- Comparaisons AE vs PCA sur test set
- Outputs : plots dans `results/`

---

### Pan 2b — Migration tempodata → MLflow (après Pan 2)

Migrer les runs déjà entraînés depuis `tempodata/` vers `mlruns/` via un script de migration (pas de réentraînement).
Structure source : `tempodata/autoencoder/{run_name}/{tag}/` contient `.pth`, `.png`, et `.txt` de métriques.
Écrire un script qui parse les noms de dossiers (params), lit les `.txt` (métriques), et crée des runs MLflow rétrospectivement.

---

### Pan 3 — Finalisation pour GitHub (après refactoring)

- README utilisateur (installation, usage, description des pipelines)
- Docstrings complets sur toutes les fonctions publiques
- Type hints
- Tests unitaires avec pytest (fonctions critiques : loader, métriques, preprocessing)
- Nettoyage final de `archive_scripts/`

---

## Environnement local
- Python 3.11.9 via pyenv
- Venv : `.venv/` (ne pas commiter)
- Dépendances : `pip install -r requirements_locked.txt`
- `.env` local à créer depuis `.env.example` (sans clés AWS pour tests locaux)