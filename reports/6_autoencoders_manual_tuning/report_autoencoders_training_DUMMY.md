# Autoencoders — Entraînement et régularisation

> **Données** : ACDC Cardiac MRI Dataset
> Citation obligatoire : O. Bernard, A. Lalande, C. Zotti, F. Cervenansky, *et al.*, "Deep Learning Techniques for Automatic MRI Cardiac Multi-structures Segmentation and Diagnosis: Is the Problem Solved?", *IEEE Transactions on Medical Imaging*, vol. 37, no. 11, pp. 2514-2525, Nov. 2018. doi: 10.1109/TMI.2018.2837502
>
> **Scripts** : `scripts/run_autoencoder.py`, `src/training/ae_training.py`
> **Config(s)** : `autoencoder.yaml` *(copie figée dans ce dossier — version vivante dans `configs/`)*
> **Voir aussi** : `4_autoencoders_architectures/` (choix d'architecture), `6_variational_autoencoders/` (extension VAE)

## Objectif

Entraîner un autoencodeur 3D capable de représenter les volumes cardiaques dans un espace latent de faible dimension, avec une qualité de reconstruction/régression comparable à la PCA spatiale (notre baseline). Ce rapport documente la procédure d'entraînement elle-même (fonction de perte, early stopping, régularisation) — pas la comparaison d'architectures, traitée dans `4_autoencoders_architectures/`.

## Démarche : une convergence progressive

Les résultats finaux ne sont pas sortis directement d'une seule configuration d'entraînement. Ils sont le produit de plusieurs itérations, chacune motivée par un problème identifié dans la précédente. Le tableau ci-dessous résume la progression ; le détail de chaque étape suit.

| Itération | Changement principal | Résultat clé | Motivation de l'étape suivante |
|---|---|---|---|
| 1 — AE linéaire de référence | Premier AE entraîné (purement linéaire), comparé à la PCA spatiale | R² validation = 0.78 (12000 epochs, 120 patients) vs R² = 0.9 pour la PCA | Écart significatif avec la PCA à expliquer avant d'investir dans des architectures non-linéaires |
| 2 — Correction des métriques | Bug identifié dans le calcul des métriques (mean subtraction dupliquée) | R² remonte de 0.5–0.6 (calcul erroné) à 0.7–0.75 (calcul corrigé) | Toujours en dessous de la PCA (0.75–0.9) → suspicion de sur-apprentissage |
| 3 — Régularisation | Ajout de dropout, weight decay, et bruit (denoising) | Dropout 0.1 améliore la généralisation ; weight decay 1e-4 a un effet marginal ; bruit (std=0.005) dégrade la performance | Garder dropout seul, écarter le denoising à ce niveau de bruit |
| 4 — *(à compléter)* | *(architecture finale + recherche d'hyperparamètres via Optuna, voir `run_ae_optuna.py`)* | *(R² final, n_runs, latent_dim retenue)* | — |

### Itération 1 — AE linéaire de référence

Avant d'investir dans des architectures profondes, un AE purement linéaire a été entraîné sur 120 patients (12000 epochs) pour vérifier qu'un autoencodeur peut au moins approcher la PCA — un AE linéaire sans biais ni non-linéarité devrait théoriquement converger vers un sous-espace équivalent à celui de la PCA. Résultat : R² validation = 0.78, contre 0.9 pour la PCA spatiale sur le même split. L'écart, bien que net, restait dans une fourchette encourageante pour poursuivre.

*(à compléter : figure `loss_train_val_linear.png`, run_id MLflow correspondant)*

### Itération 2 — Correction des métriques

Une revue du code de calcul des métriques a révélé une double soustraction de moyenne (mean subtraction appliquée à la fois en amont et par `sklearn`). Une fois corrigée, les R² mesurés sont passés de 0.5–0.6 à 0.7–0.75 — un saut qui aurait pu être interprété à tort comme une amélioration de modèle si la cause n'avait pas été tracée jusqu'au bug de calcul. Ça reste un point important à mentionner : la performance réelle des modèles n'avait pas changé, seule la mesure était fausse.

### Itération 3 — Régularisation

Avec des métriques fiables, l'écart restant avec la PCA (0.7–0.75 vs 0.75–0.9) a été attribué à un sur-apprentissage. Trois leviers de régularisation ont été testés manuellement :

- **Dropout (0.1)** : amélioration visible de la généralisation
- **Weight decay (1e-4)** : effet marginal
- **Denoising (bruit gaussien, std=0.005)** : dégrade la performance — le bruit injecté était trop fort relativement à l'échelle des intensités voxel

*(à compléter : tableau ou figure comparant les runs avec/sans chaque régularisation, run_ids MLflow)*

### Itération 4 — *(à compléter)*

*(Décrire ici la configuration finale retenue : architecture, hyperparamètres issus d'Optuna, n_val=0 vs early stopping, et le R² final obtenu. Lien vers `7_agentic_ai_training/` si la recherche d'hyperparamètres a été assistée par un agent.)*

## Résultat final

| Méthode | R² validation | Notes |
|---|---|---|
| PCA spatiale | 0.75–0.9 | Baseline |
| AE linéaire (itération 1) | 0.78 | Avant régularisation |
| AE *(architecture finale)* | *(à compléter)* | Après régularisation + tuning Optuna |

*(à compléter : figure de la courbe R² vs dimension latente, comparant PCA et AE final — voir `run_comparison_aepca.py`)*

## Limites et perspectives

- L'AE reste, à ce stade, en dessous de la PCA spatiale sur ce jeu de données — à interpréter : est-ce intrinsèque (la PCA capture déjà l'essentiel de la variance linéaire utile) ou un manque d'exploration d'architecture/hyperparamètres ?
- *(à compléter selon l'état actuel : pistes non testées, prochaines étapes)*

## Reproduction

```bash
python scripts/run_autoencoder.py
```

Mode LOAD (recharger un modèle déjà entraîné plutôt que de relancer l'entraînement) : mettre `recalculate_ae: false` et `load_run_id: <mlflow_run_id>` dans `autoencoder.yaml`.