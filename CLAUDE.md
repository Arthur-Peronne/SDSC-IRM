# CLAUDE.md — SDSC Arthur Péronne

## Projet
Analyse de données IRM cardiaques 3D. Le pipeline va du preprocessing des images (recalage, rééchantillonnage) jusqu'à la comparaison de méthodes de réduction de dimensionnalité (PCA spatiale, autoencodeurs 3D) et régression sur métadonnées patients.

Les données médicales brutes sont sur S3 (AWS) et accessibles uniquement via Renku. Le développement se fait en local, les tests sur données réelles se font sur Renku après commit/push.

## Commandes utiles

```bash
# MLflow UI en local (depuis la racine du projet, venv activé)
mlflow ui --backend-store-uri mlruns/
# → http://127.0.0.1:5000

# MLflow UI sur Renku : ne fonctionne pas complètement
mlflow ui --host 0.0.0.0 --port 5000 
# puis cliquer sur le lien...
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
- **Autoencodeurs 3D** : plusieurs architectures (`AE3dFCDeep`, `AE3dLinear`...), sans val set (n_val=0) ou avec early stopping, Optuna pour hyperparamètres
- **Régression** : logistique et linéaire sur coordonnées PCA + AE + métadonnées patients (group, height, weight)
- **Comparaisons AE vs PCA** : mêmes splits train/val/test, lecture directe depuis MLflow

## Conventions scientifiques importantes
- **Split régression** : n_train=100, n_val=0, n_test=50. Les AE entraînés avec n_val=20 sont compatibles (mêmes 100 premiers patients). Les PCA entraînées avec n_train=120 ne le sont pas — réentraîner en 100/50.
**Stratification** : pour que les modèles PCA/AE soient utilisables en régression sur les groupes ACDC, les entraîner sur un split stratifié (5 groupes équilibrés). Le splitdefault est par chance stratifié (ignoré de toute façon par `get_split_indices` quand `special_split=null`) ; pour les splits seedés, mettre `stratify_ongroup: true` dans le YAML (PCA et AE).
- **AE sans val set** : n_val=0 est désormais la norme. `run_regression.py` gère le fallback `best_epoch → n_epochs` si le param n'est pas logué.
- **PCA artifact** : sauvegardé comme `pca_{frame_tag}.joblib` (ex: `pca_ED.joblib`, `pca_ED+ES.joblib`) dans les artifacts MLflow du run `pca_spatial`.

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

**Étape 7 : PCA spatiale** ✅ — `run_pca_spatial.py`
- `run_pca_spatial.py` réécrit : `main()`, YAML, MLflow (`start_run`/`resume_run`), `loader.load_numpy_splits()`
- `src/models/pca_spatial.py` : `TEMPODATA_FOLDER` → `PROCESSED_IMAGES_FOLDER`, `plot_eigenvectors` prend `nii_ref` en param, `pca_compute_metrics` simplifié, `pca_patients()` supprimé (dead code)
- `src/models/pca.py` : ajout `pca_spatial_reconstruct` (reconstruction 3D spatiale, partagée)
- `configs/pca_spatial.yaml` : `cache_folder`, `load_run_id`, `pc_max`, `eigenvectors_to_plot` liste, `n_pc_to_reconstruct`
- Harmonisation temporel/spatial : flags `plot_*` unifiés dans les deux YAML/scripts
- Row centering (par patient) explicite avant sklearn, row_means conservés pour reconstruction
- Métriques R²/MSE/MAE/RMSE par dimension latente logées dans MLflow (`step=latent_dimensions` → courbe R² vs n_PCs dans l'UI) via `tracking.log_metric`
- Testé et validé sur Renku

**Étape 7bis : Splits seeds pour PCA spatiale** ✅ — `run_pca_spatial.py` + `src/data/splits.py`
- `src/data/splits.py` (nouveau) : `get_split_indices(n_train, n_val, n_test, special_split, stratify, n_patients=150)` — séquentiel ("splitdefault") ou seedé, stratification optionnelle
- `src/data/loader.py` : `_split_frames` utilise indices numpy, `load_numpy_splits` + `load_tensor_datasets` nouvelle signature, retournent `split_name`
- `configs/pca_spatial.yaml` : `n_train`, `n_val`, `n_test`, `special_split: null`
- `run_pca_spatial.py` : données chargées avant ouverture MLflow run (pour avoir split_name dans run_name), vérification cohérence split au LOAD
- `src/config.py` : RESULTS_FOLDER + PROCESSED_IMAGES_FOLDER créés avec mkdir à l'import
- Testé et validé sur Renku

**Étape 7ter : Fusion run_registration_pipelinechecks dans run_registration** ✅ — `run_registration.py` + `configs/registration.yaml`
- `run_pipelinechecks: false` ajouté dans `registration.yaml` (+ `check_ED`, `check_ES`, `check_resampled`, `check_cropped`, `check_registered`, `do_dice_checks`, `registered_OLD`, `n_worst_to_print`, `patients_ED`, `patients_ES`)
- Logique de `run_registration_pipelinecheck.py` déplacée comme section conditionnelle en fin de `run_registration.py`
- Pour lancer les checks seuls : mettre `resample_all: false`, `crop_all: false`, `register_all: false`, `run_pipelinechecks: true`
- Testé et validé sur Renku

**Étape 8 : Autoencoder** ✅ — `run_autoencoder.py` + `ae_training.py`
- `run_autoencoder.py` réécrit : lit YAML, `loader.load_tensor_datasets()`, MLflow `start_run`/`resume_run`, `log_params`/`log_metric`/`log_model_state_dict`
- LOAD mode via `load_run_id` + vérification cohérence split/modèle (même pattern que PCA spatiale)
- `ae_training.py` nettoyé : dead code supprimé (`ae_training_old`, `ae_getdataset`, `dataset_for_metrics`), fonctions pures (plus de I/O fichier), `get_best_epochs_stats_from_mlflow` via MLflow
- `ae_plots.py` : `plot_train_val_loss` sauvegarde dans RESULTS_FOLDER et retourne le path (loggable MLflow)
- `loader.load_tensor_datasets` : ajout param `frame_type` pour cohérence avec `load_numpy_splits`
- Testé et validé sur Renku
- `mask_ys`/`mask_bin` dans `autoencoder.yaml` non encore connectés à `load_tensor_datasets` (hardcodé `mask=False, binary_mask=False`) → Pan 4
- Fonctions de comparaison dans `ae_plots.py` lisent encore TEMPODATA_FOLDER → à migrer en Étape 10

**Étape 9 : Hyperparamètres Optuna** ✅ — `run_ae_optuna.py` + `src/training/ae_optuna.py`
- `run_ae_optuna.py` réécrit (renommé depuis `run_ae_hyperparam.py`) : lit `configs/ae_optuna.yaml`, `loader.load_tensor_datasets()`, MLflow via `start_run_and_get_id`/`resume_run`
- Deux modes : CALC (`plot_only: false`) lance l'étude Optuna et persiste la DB SQLite dans l'artifact dir MLflow ; PLOT (`plot_only: true` + `load_run_id`) recharge la DB via `tracking.download_artifact` et regénère les plots
- `src/training/ae_optuna.py` : aucun `TEMPODATA_FOLDER`, DB path passé en paramètre, fonctions pures — `run_optuna`, `load_study`, `plot_optuna_results` (3 graphiques : history, HP evolution, HP vs loss), `save_optuna_summary`
- `configs/ae_optuna.yaml` : ranges de recherche par type (`float`/`int`, log scale), trial initial enqueuable (`hp_initial`), support VAE (`beta`, `beta_warmup_epochs`)
- Params fixes (non optimisés) loggés séparément dans MLflow ; meilleurs params loggés comme métriques (`best_lr`, etc.)
- Des runs validés existent dans MLflow (expérience `ae_optuna`, `mlruns/318366704909602552/`)

**Étape 10 : Comparaisons AE vs PCA** ✅ — `run_comparison_aepca.py` + `src/visualization/ae_plots.py`
- `run_comparison_aepca.py` réécrit : lit `configs/comparison_aepca.yaml`, lecture MLflow via `search_runs()` + `get_metric_history()`, outputs dans `results/` (pas de tracking MLflow)
- Un seul plot générique configurable via une liste de `series` (model_name + experiment_tag + `params_filter` optionnel)
  - `model_name: "PCA"` → expérience `pca_spatial` (une courbe continue, step = n_pcs)
  - tout autre model_name → expérience `autoencoder` (un run par latent_dim, agrégés automatiquement)
- `params_filter` : dict de params MLflow supplémentaires pour discriminer quand `model_name + experiment_tag` ne suffisent pas (ex : `frame_tag`, `n_train`)
- `plotname` : null = nom auto-généré depuis les labels des séries ; string = nom de fichier custom
- `superpose_metrics` : false = un subplot par métrique ; true = toutes sur un seul axes (couleur = série, linestyle + linewidth = métrique)
- WARNINGs imprimés (jamais bloquants) si params incohérents au sein d'une série ou entre séries
- `run_stats_nepochs` : stats best_epoch depuis MLflow directement (sans `split_name` obligatoire)
- `plot_comparison_curves()` ajoutée dans `ae_plots.py` ; anciennes fonctions txt-based conservées (dead code, nettoyage Pan 4)
- Validé en local sur runs MLflow existants

**Étape 11 : Régression** ✅ — `run_regression.py`
- `scripts/run_regression.py` réécrit : YAML, MLflow `start_run`, métriques à `step=n_dims`
- `src/models/regression.py` : fonctions pures (fit_scaler, fit_logistic, fit_linear, eval_*)
- `src/visualization/regression_plots.py` : plots in-memory (logistic metrics, linear metrics, confusion matrix, predicted vs true)
- `src/data/importdata.py` : `load_patient_metadata(y_name)` + `load_acdc_groups` comme wrapper
- `configs/regression.yaml` : source_type (pca/ae), y_name, n_train=100, cumvar_threshold_list, ae_source, plot_only/load_run_id, logistic_C, n_pc_confusion
**Source PCA** : charge la PCA depuis MLflow (`pca_{frame_tag}.joblib`), sweep sur `latent_dims_list` (liste fixe de n_pc, partagée avec les AE pour comparabilité), row-centering identique à `run_pca_spatial.py`
- **Source AE** : `search_runs()` par model_name/experiment_tag/split_name/params_filter, groupés par latent_dim, vérification split sur tous les runs, encoding via `collect_latent_vectors`
- **Vérification split** : `split_name + n_train` comparés entre le run PCA/AE source et la config régression — ValueError si incohérence
- **Split régression** : n_train=100, n_test=50, pas de val set. Compatible avec AE entraînés en 100/20/30 (mêmes 100 premiers patients). **Incompatible avec PCA entraînées en 120/30** → réentraîner la PCA spatiale en 100/50 avant de lancer la régression PCA
- **JSON artifacts** : confusion matrix + Y_pred sauvegardés en `.json` dans `mlruns/` (committés git) → `plot_only: true` + `load_run_id` régénère tous les plots sans données ni modèles
- `logistic_C` : null = auto (0.5 si ED+ES pour corriger les paires corrélées, 1.0 sinon)
- Y dupliqué si `use_both_frames` (dérivé de `frame_tag == "ED+ES"` depuis les params MLflow du run source)

**Stratification (`stratify_ongroup`)** ✅
- `configs/pca_spatial.yaml` et `configs/autoencoder.yaml` : nouveau param `stratify_ongroup` (true = stratifie le split sur le groupe ACDC, ignoré si `special_split: null`)
- `src/data/loader.py` : `load_numpy_splits`/`load_tensor_datasets` prennent `stratify_ongroup`, construisent le tableau de groupes via `load_patient_metadata("group", n_patients)` et le passent à `get_split_indices(stratify=...)`
- Loggé comme param MLflow par `run_pca_spatial.py`, `run_autoencoder.py`, `run_ae_optuna.py`
- `run_regression.py` : lit `stratify_ongroup` depuis les params du run PCA/AE source et le repropage à `loader`/`_apply_split_to_Y` pour reproduire exactement les mêmes indices (split nommé + stratify_ongroup déterminent ensemble les indices, pas le nom seul)
- Testé et validé sur Renku (split "split2" stratifié → 10 patients/groupe en test, soit 20 frames/groupe en ED+ES)
---

**Comparaison multi-splits (`compare_mode`)** ✅ — `run_regression.py`
- `configs/regression.yaml` : `compare_mode: true` + `load_run_ids: [...]`
- Charge les JSON `results_{n}pc/latdim.json` de plusieurs runs de régression (splits différents), sélectionne `n_pc_confusion` ou la dimension la plus proche (échelle log, WARNING si différent)
- `_check_consistent_runs` : WARNING (non bloquant) si `n_train`/`source_type`/`ae_model_name`/`ae_experiment_tag` diffèrent entre les runs sélectionnés
- `plot_average_confusion_matrix` (nouveau, `regression_plots.py`) : moyenne des matrices normalisées par ligne + intervalle de confiance de Wilson par cellule (bornes `[lo,hi]`, comptes poolés sur tous les splits)
- `plot_confusion_matrix` (single-split) : même format d'IC de Wilson, sur les comptes du split unique
- Sauvegardé directement dans `RESULTS_FOLDER` (pas de tracking MLflow) : `confusionmatrix_average_{source_type}_{n_splits}splits_{n}{pc|latdim}_{experiment_tag}.png`

### Pan 3 — Finalisation pour GitHub (après refactoring)

- README utilisateur (installation, usage, description des pipelines)
- Docstrings complets sur toutes les fonctions publiques
- Type hints
- Tests unitaires avec pytest (fonctions critiques : loader, métriques, preprocessing)
- Nettoyage final de `archive_scripts/`

---

### Pan 4 — Nettoyage et extensions mineures (après validation Pan 2)

- `mask_ys`/`mask_bin` dans `autoencoder.yaml` non encore connectés à `load_tensor_datasets` (hardcodé `mask=False, binary_mask=False`) — ajouter les params quand nécessaire, copier le pattern de `load_numpy_splits`
- Fonctions de comparaison dans `ae_plots.py` (`plot_ae_comparison`, `plot_ae_vs_pca`, etc.) lisent encore depuis TEMPODATA_FOLDER via `_load_summarymetrics` — remplacées fonctionnellement par `plot_comparison_curves`, à supprimer lors du nettoyage final
- Supprimer `experiments/archive_scripts/` (nettoyage final Pan 3)

---

## Environnement local
- Python 3.11 (3.11.15 sur Fedora via `dnf install python3.11`, 3.11.9 sur macOS via pyenv)
- Venv : `.venv/` (ne pas commiter) — **toujours créer avec `python3.11 -m venv .venv`** (pas `python3` ni `python`, qui pointent vers le Python système)
- Dépendances : `pip install -r requirements_locked.txt`
- `.env` local à créer depuis `.env.example` (sans clés AWS pour tests locaux)
- lancer setup_env.sh pour installer le présent package