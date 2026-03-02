# Projet Final - Reduction de dimension

## Structure
- `notebooks/pca.ipynb`: notebook PCA (projection 2D, visualisation, observation, export).
- `generate.py`: genere les sorties 2D dans `outputs/` (actuellement PCA).
- `evaluate.py`: compare les methodes via la metrique `trustworthiness`.
- `data/city_lifestyle_dataset.csv`: donnees source.

## Prerequis
- Python 3.10+
- Dependances:

```bash
pip install pandas scikit-learn matplotlib numpy
```

## Execution
1. Generer les sorties 2D:

```bash
python generate.py
```

2. Evaluer les methodes disponibles:

```bash
python evaluate.py
```

## Convention pour la comparaison
- Le script `evaluate.py` lit tous les fichiers `*_2d.csv` dans `outputs/`.
- Exemples attendus:
  - `outputs/pca_2d.csv`
  - `outputs/tsne_2d.csv`
  - `outputs/umap_2d.csv`

Chaque fichier doit contenir au moins 2 colonnes numeriques (coordonnees 2D) et le meme nombre de lignes que les donnees source.

## Notes Git
- Les fichiers `outputs/*.csv` sont ignores par `.gitignore`.
- Les sorties peuvent etre regenerees localement avant l'evaluation.
