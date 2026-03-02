# Projet Final - Reduction de dimension

## Structure
- `notebooks/pca.ipynb`: notebook PCA (projection 2D, visualisation, observation, export).
- `generate.py`: genere les sorties 2D dans `outputs/` (PCA + t-SNE).
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

Options:

```bash
python generate.py --methods pca
python generate.py --methods tsne
python generate.py --methods pca tsne
```

2. Evaluer les methodes disponibles:

```bash
python evaluate.py
```

3. Evaluer seulement certains modeles (bonus):

```bash
python evaluate.py --methods pca
python evaluate.py --methods pca tsne umap
```

## Docker
1. Build et execution standard:

```bash
docker build -t cont-26-eval .
docker run --rm cont-26-eval
```

2. Bonus volume (mettre a jour code/donnees apres dockerisation):

```bash
docker run --rm -v ${PWD}:/app cont-26-eval
```

Avec Docker Compose:

```bash
docker compose up --build
```

Le volume `.:/app` permet de modifier les scripts (`evaluate.py`, `generate.py`) ou les donnees (`data/`) localement, puis relancer le conteneur sans reconstruire l'image a chaque changement.

## Workflow local collaboratif avec Docker (bonus +2)
Objectif: travailler tous sur la meme version de Python et des librairies, tout en modifiant le code/donnees localement.

1. Build de l'environnement commun (une seule fois ou apres changement des dependances):

```bash
docker compose build
```

2. Lancer les scripts avec le projet monte en volume:

```bash
docker compose run --rm app sh -c "python generate.py && python evaluate.py"
```

3. Tester seulement certains modeles:

```bash
docker compose run --rm app python evaluate.py --methods pca
docker compose run --rm app python evaluate.py --methods pca tsne umap
```

4. Boucle de developpement locale:
- modifier `evaluate.py`, `generate.py`, `data/*` ou d'autres fichiers localement
- relancer `docker compose run --rm app ...`
- aucun rebuild n'est necessaire tant que `requirements.txt` ne change pas

5. Si les dependances changent:

```bash
docker compose build --no-cache
```

Ce processus garantit un environnement identique pour toute l'equipe (meme image Docker), avec iteration rapide grace au volume `.:/app`.

## Convention pour la comparaison
- Le script `evaluate.py` lit tous les fichiers `*_2d.csv` dans `outputs/`.
- Exemples attendus:
  - `outputs/pca_2d.csv`
  - `outputs/tsne_emb_2d.csv` (ou `outputs/tsne_2d.csv`)
  - `outputs/umap_2d.csv`

Chaque fichier doit contenir au moins 2 colonnes numeriques (coordonnees 2D) et le meme nombre de lignes que les donnees source.

## Notes Git
- Les fichiers `outputs/*.csv` sont ignores par `.gitignore`.
- Les sorties peuvent etre regenerees localement avant l'evaluation.
