from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap, TSNE
from sklearn.preprocessing import StandardScaler


def load_numeric_data(data_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(data_path)
    numeric = df.select_dtypes(include=["number"]).copy()
    if numeric.empty:
        raise ValueError("No numeric columns found in original dataset.")
    if numeric.isna().any().any():
        raise ValueError("Missing values found in numeric columns.")
    return df, numeric


def export_pca_2d(data_path: Path, outputs_dir: Path, random_state: int) -> Path:
    df, numeric = load_numeric_data(data_path)
    x_scaled = StandardScaler().fit_transform(numeric)
    x_2d = PCA(n_components=2, random_state=random_state).fit_transform(x_scaled)

    out_df = pd.DataFrame(x_2d, columns=["PC1", "PC2"])
    if "city_name" in df.columns:
        out_df["city_name"] = df["city_name"].values

    outputs_dir.mkdir(parents=True, exist_ok=True)
    out_path = outputs_dir / "pca_2d.csv"
    out_df.to_csv(out_path, index=False)
    return out_path


def export_tsne_2d(data_path: Path, outputs_dir: Path, random_state: int) -> Path:
    df, numeric = load_numeric_data(data_path)
    x_scaled = StandardScaler().fit_transform(numeric)
    x_2d = TSNE(
        n_components=2,
        perplexity=30,
        random_state=random_state,
        init="pca",
        learning_rate="auto",
    ).fit_transform(x_scaled)

    out_df = pd.DataFrame(x_2d, columns=["tsne_1", "tsne_2"])
    if "city_name" in df.columns:
        out_df["city_name"] = df["city_name"].values

    outputs_dir.mkdir(parents=True, exist_ok=True)
    out_path = outputs_dir / "tsne_emb_2d.csv"
    out_df.to_csv(out_path, index=False)
    return out_path


def export_umap_2d(data_path: Path, outputs_dir: Path, random_state: int) -> Path:
    try:
        from umap import UMAP
    except ImportError as exc:
        raise ImportError(
            "UMAP export requires 'umap-learn'. Install dependencies from requirements.txt."
        ) from exc

    df, numeric = load_numeric_data(data_path)
    x_scaled = StandardScaler().fit_transform(numeric)
    x_2d = UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        random_state=random_state,
    ).fit_transform(x_scaled)

    out_df = pd.DataFrame(x_2d, columns=["umap_1", "umap_2"])
    if "city_name" in df.columns:
        out_df["city_name"] = df["city_name"].values

    outputs_dir.mkdir(parents=True, exist_ok=True)
    out_path = outputs_dir / "umap_2d.csv"
    out_df.to_csv(out_path, index=False)
    return out_path


def export_isomap_2d(data_path: Path, outputs_dir: Path) -> Path:
    df, numeric = load_numeric_data(data_path)
    x_scaled = StandardScaler().fit_transform(numeric)
    x_2d = Isomap(
        n_components=2,
        n_neighbors=10,
    ).fit_transform(x_scaled)

    out_df = pd.DataFrame(x_2d, columns=["isomap_1", "isomap_2"])
    if "city_name" in df.columns:
        out_df["city_name"] = df["city_name"].values

    outputs_dir.mkdir(parents=True, exist_ok=True)
    out_path = outputs_dir / "isomap_emb_2d.csv"
    out_df.to_csv(out_path, index=False)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate exported 2D files used by the comparison script."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data") / "city_lifestyle_dataset.csv",
        help="Path to original dataset (default: data/city_lifestyle_dataset.csv).",
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs"),
        help="Output directory for generated 2D CSVs (default: outputs/).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducible generation (default: 42).",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["pca", "tsne", "umap", "isomap"],
        help="Methods to export (default: pca tsne umap isomap).",
    )
    args = parser.parse_args()

    if not args.data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {args.data_path.resolve()}")

    selected = {m.strip().lower() for m in args.methods if m.strip()}
    generated: list[Path] = []

    if "pca" in selected:
        generated.append(
            export_pca_2d(
                data_path=args.data_path,
                outputs_dir=args.outputs_dir,
                random_state=args.random_state,
            )
        )
    if "tsne" in selected:
        generated.append(
            export_tsne_2d(
                data_path=args.data_path,
                outputs_dir=args.outputs_dir,
                random_state=args.random_state,
            )
        )
    if "umap" in selected:
        generated.append(
            export_umap_2d(
                data_path=args.data_path,
                outputs_dir=args.outputs_dir,
                random_state=args.random_state,
            )
        )
    if "isomap" in selected:
        generated.append(
            export_isomap_2d(
                data_path=args.data_path,
                outputs_dir=args.outputs_dir,
            )
        )

    if not generated:
        raise ValueError(
            "No supported method selected. Use any of: pca, tsne, umap, isomap."
        )

    for path in generated:
        print(f"Generated: {path}")


if __name__ == "__main__":
    main()
