from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.manifold import trustworthiness
from sklearn.preprocessing import StandardScaler


def load_original_numeric_data(data_path: Path) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    numeric = df.select_dtypes(include=["number"]).copy()
    if numeric.empty:
        raise ValueError("No numeric columns found in original dataset.")
    if numeric.isna().any().any():
        raise ValueError("Missing values found in original numeric data.")
    return numeric


def load_embedding_2d(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if len(numeric_cols) < 2:
        raise ValueError(f"{csv_path.name}: expected at least 2 numeric columns.")
    return df[numeric_cols[:2]]


def method_name_from_file(csv_path: Path) -> str:
    name = csv_path.stem.lower()
    return name[:-3] if name.endswith("_2d") else name


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare dimensionality-reduction methods using trustworthiness "
            "from exported 2D CSV files."
        )
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
        help="Directory containing method exports named *_2d.csv (default: outputs/).",
    )
    parser.add_argument(
        "--neighbors",
        type=int,
        default=5,
        help="Number of neighbors for trustworthiness (default: 5).",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help=(
            "Optional list of methods to evaluate (e.g. --methods pca tsne). "
            "Method names are inferred from filenames like <method>_2d.csv."
        ),
    )
    args = parser.parse_args()

    if not args.data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {args.data_path.resolve()}")
    if not args.outputs_dir.exists():
        raise FileNotFoundError(f"Outputs directory not found: {args.outputs_dir.resolve()}")

    x_numeric = load_original_numeric_data(args.data_path)
    x_scaled = StandardScaler().fit_transform(x_numeric)

    exported_files = sorted(args.outputs_dir.glob("*_2d.csv"))
    if args.methods:
        selected = {m.strip().lower() for m in args.methods if m.strip()}
        exported_files = [
            p for p in exported_files if method_name_from_file(p).lower() in selected
        ]
    if not exported_files:
        raise FileNotFoundError(
            f"No matching exported 2D files found in {args.outputs_dir.resolve()} (expected *_2d.csv)."
        )

    results: list[tuple[str, float, str]] = []
    for file_path in exported_files:
        method = method_name_from_file(file_path)
        try:
            embedding_2d = load_embedding_2d(file_path)
            if len(embedding_2d) != len(x_scaled):
                raise ValueError(
                    f"Row count mismatch (embedding={len(embedding_2d)}, original={len(x_scaled)})."
                )
            score = trustworthiness(
                x_scaled,
                embedding_2d.to_numpy(),
                n_neighbors=args.neighbors,
            )
            results.append((method, score, "ok"))
        except Exception as exc:  # Keep per-method reporting robust.
            results.append((method, float("nan"), f"error: {exc}"))

    print("Trustworthiness comparison")
    print(f"Dataset: {args.data_path}")
    print(f"Neighbors: {args.neighbors}")
    print("-" * 54)
    print(f"{'Method':<15}{'Score':<12}{'Status'}")
    print("-" * 54)
    for method, score, status in sorted(results, key=lambda r: (r[1] != r[1], -r[1] if r[1] == r[1] else 0)):
        score_str = f"{score:.6f}" if score == score else "N/A"
        print(f"{method:<15}{score_str:<12}{status}")


if __name__ == "__main__":
    main()
