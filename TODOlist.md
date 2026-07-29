### Pan 3 — Finalisation pour GitHub (après refactoring)

- README utilisateur (installation, usage, description des pipelines)
- Docstrings complets sur toutes les fonctions publiques
- Type hints
- Tests unitaires avec pytest (fonctions critiques : loader, métriques, preprocessing)

---

### Pan 4 — Nettoyage et extensions mineures

- `mask_ys`/`mask_bin` dans `autoencoder.yaml` non encore connectés à `load_tensor_datasets` (hardcodé `mask=False, binary_mask=False`) — ajouter les params quand nécessaire, copier le pattern de `load_numpy_splits`
- Fonctions de comparaison dans `ae_plots.py` (`plot_ae_comparison`, `plot_ae_vs_pca`, etc.) lisent encore depuis TEMPODATA_FOLDER via `_load_summarymetrics` — remplacées fonctionnellement par `plot_comparison_curves`, à supprimer lors du nettoyage final
