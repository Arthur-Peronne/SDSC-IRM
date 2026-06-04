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
- **mlruns/** : métadonnées MLflow (meta.yaml, metrics, params, tags) **committées dans git** pour pouvoir utiliser `mlflow ui` en local. Les artefacts (modèles `.pth`, etc.) sont exclus via `.gitignore` (`mlruns/**/artifacts/`). Workflow : entraîner sur Renku → commit + push mlruns/ → pull en local → `mlflow ui`
- **processed_images/** : fichiers nii générés (coûteux), rechargés manuellement entre sessions Renku — **ne va pas dans git**

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

**Étape 4 : Visualisation MRI** ✅
- `configs/visualization.yaml` : patient, source (raw/processed), frame_type (ED/ES/both), plot_modes
- `run_visualize.py` réécrit : lit Info.cfg pour les numéros de frames, construit les chemins, appelle `mri_plots`
- `src/visualization/mri_plots.py` : `plot_masks` supprimée (redondante), `plot_allepochs` harmonisé avec `plot_anat`
- Modes : image / mask / overlay / all_epochs — naming cohérent `patientXXX_frameXX_{source}_{suffix}_{frame_type}`
- Testé sur Renku (raw ED/ES, processed registered, 4D)

**Étape 5 : Image registration** ✅ — `run_registration.py` + `run_registration_pipelinechecks.py`
- `configs/registration.yaml` enrichi : tous les paramètres exposés (target_spacing, crop_shape, reference_patient/frame, n_iterations, limit)
- `run_registration.py` réécrit : lit le YAML, passe tous les paramètres aux fonctions
- Migration `TEMPODATA_FOLDER` → `PROCESSED_IMAGES_FOLDER` dans tous les modules `src/data/` et scripts
- `registered_framesBIS` supprimé partout, `registered_frames` comme seule référence
- `mkdir` ajouté dans `resample_all` et `crop_all_frames` (manquait)
- Suivi d'avancement patient par patient dans `resample_all` et `crop_all_frames`
- `load_dotenv` ancré sur `__file__` dans `config.py` (lancement depuis n'importe quel répertoire)
- `run_registration_pipelinechecks.py` : comparaison OLD réécrite (ED+ES, lookup par nom, vérification de couverture)
- `number_of_iterations` exposé dans `register_all_frames`
- Testé sur Renku : 300 frames (ED+ES), DICE after 0.723 (mean), 282/300 améliorés

**Étape 6 : PCA temporelle** ✅ — `run_pca_temporal.py`
- `configs/pca_temporal.yaml` : patient_id, recalculate_pca, max_pc_calc, pc_max, eigenvectors_to_plot, frames_to_reconstruct
- `run_pca_temporal.py` réécrit : lit YAML, MLflow via `start_run` / `resume_run`, chargement via `get_patient_acdc_path`
- Suppression du double centrage (sklearn PCA centre déjà par colonne)
- `src/models/pca.py` : ajout `eigenvector_to_nii` et `pca1_reconstruct` (retournent des nii, aucun plot)
- `src/tracking.py` : ajout `log_sklearn_model` (joblib) et `resume_run` (rouvre un run existant pour ajouter des artifacts)
- Plots via `mrp.plot_oneimg` / `mrp.plot_allepochs` — identiques à `run_visualize`
- Image moyenne (0 PC) plottée dans la reconstruction
- Testé sur Renku : patients training et testing, CALC et LOAD, cas limites

**Étape 7 : PCA spatiale** 🔄 en test sur Renku — `run_pca_spatial.py`
- `run_pca_spatial.py` réécrit : `main()`, YAML, MLflow (`start_run`/`resume_run`), `loader.load_numpy_splits()`
- `src/models/pca_spatial.py` : `TEMPODATA_FOLDER` → `PROCESSED_IMAGES_FOLDER`, `plot_eigenvectors` prend `nii_ref` en param, `pca_compute_metrics` simplifié
- `src/models/pca.py` : ajout `pca_spatial_reconstruct` (reconstruction 3D spatiale, partagée)
- `configs/pca_spatial.yaml` : `cache_folder`, `load_run_id`, `pc_max`, `eigenvectors_to_plot` liste, `n_pc_to_reconstruct`
- Harmonisation temporel/spatial : flags `plot_*` unifiés dans les deux YAML/scripts
- Row centering (par patient) explicite avant sklearn, row_means conservés pour reconstruction
- Métriques R²/MSE/MAE/RMSE par dimension latente logées dans MLflow (`step=latent_dimensions` → courbe R² vs n_PCs dans l'UI) via `tracking.log_metric`
- **Pending après validation Renku** :
  - Supprimer `pca_patients()` (dead code dans `pca_spatial.py`) et `archive_scripts/run_pca_spatial_ARCHIVED.py`
  - `ae_aggregate_metrics` pointe encore vers `TEMPODATA_FOLDER` pour les runs AE (`ae=True`) — à migrer dans Étape 8

**Étape 7bis : Splits seeds pour PCA spatiale** — `run_pca_spatial.py` + `src/data/splits.py`
- Extraire `get_or_create_split_indices` + `splitname_to_seed` de `regression.py` vers `src/data/splits.py`
- Mettre à jour `loader.py` pour utiliser ce mécanisme
- `run_pca_spatial.py` : remplacer le split séquentiel (`n_development`) par `splitname → seed`
- `configs/pca_spatial.yaml` : remplacer `n_development`/`n_validation` par `splitname`, `n_train`, `n_test`
- Objectif : pouvoir lancer la PCA sur de nombreux splits différents et moyenner les résultats de régression

**Étape 8 : Autoencoder** — `run_autoencoder.py` + `run_ae_hyperparam.py`
- Utilise `loader.py` unifié avec splits seeds (même mécanisme que PCA spatiale après Étape 7bis)
- Split par défaut : séquentiel (patients 1-100 train, 101-120 val, 121-150 test) ou `splitname → seed`
- Entraînement AE (plusieurs architectures, early stopping)
- Optuna pour optimisation hyperparamètres
- Métriques (R², etc.) sur train/val/test logées dans MLflow
- `ae_aggregate_metrics` : migrer de `TEMPODATA_FOLDER` vers MLflow
- Plots (loss history, reconstructions)

**Étape 9 : Comparaisons AE vs PCA** — `run_comparison_aepca.py`
- **⚠ Vérification préliminaire obligatoire** : AE et PCA comparés doivent être entraînés sur les mêmes splits (vérifier `splitname` dans les params MLflow)
- Lecture directe depuis MLflow via `search_runs()` (remplace le parsing de `.txt`)
- Comparaisons AE-AE (architectures, ED vs ED+ES, baseline vs Optuna)
- Comparaisons AE vs PCA sur test set
- Outputs : plots dans `results/`
- **Note métriques** : courbes R² vs latent dims via `get_metric_history(run_id, key)` ; dédupliquer par step en gardant le timestamp le plus récent

**Étape 10 : Régression** — `run_regression.py`
- From PCA results (refactoring de l'existant)
- From AE results (à écrire)
- **⚠ Vérification préliminaire** : si comparaison AE vs PCA, vérifier que les splits sont identiques
- Utilise `src/data/splits.py` (partagé avec PCA/AE) — permet de moyenner sur de nombreux splits pour réduire la stochasticité
- Métriques (accuracy, R², confusion matrix) + plots
- Outputs dans MLflow

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