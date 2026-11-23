# Load and describe the UCI Wine dataset.
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "wine.data"
FIG_DIR = ROOT / "figures"

COLUMNS = [
    "cultivar", "alcohol", "malic_acid", "ash", "alcalinity_of_ash",
    "magnesium", "total_phenols", "flavanoids", "nonflavanoid_phenols",
    "proanthocyanins", "color_intensity", "hue", "od280_od315", "proline",
]


def load_wine(path=DATA_FILE):
    df = pd.read_csv(path, header=None, names=COLUMNS)
    y = df["cultivar"]
    X = df.drop(columns="cultivar")
    return X, y


def describe(X, y):
    n, p = X.shape
    print(f"Rows (n): {n}    Columns (p): {p}\n")

    print("Variable types:")
    print(X.dtypes.to_string(), "\n")

    missing = X.isna().sum()
    print(f"Missing values: {missing.sum()} in total")
    if missing.sum() > 0:
        print(missing[missing > 0].to_string())
    print()

    print("Cultivar counts (not used in PCA):")
    print(y.value_counts().sort_index().to_string(), "\n")

    summary = pd.DataFrame({
        "mean": X.mean(),
        "std (s_j)": X.std(ddof=1),
        "min": X.min(),
        "max": X.max(),
    })
    print("Summary statistics:")
    print(summary.round(3).to_string(), "\n")

    s = X.std(ddof=1)
    print(f"Largest s_j:  {s.idxmax()} = {s.max():.3f}")
    print(f"Smallest s_j: {s.idxmin()} = {s.min():.3f}")
    print(f"Ratio max/min: {s.max() / s.min():.1f}")
    print(f"All s_j > 0: {bool((s > 0).all())}\n")

    var_share = (s**2) / (s**2).sum()
    print("Share of tr(C) per variable (unstandardized):")
    print(var_share.sort_values(ascending=False).round(4).to_string(), "\n")

    Z = (X - X.mean()) / s
    outliers = (Z.abs() > 3).sum()
    print(f"Entries with |z| > 3: {outliers.sum()} in total")
    if outliers.sum() > 0:
        print(outliers[outliers > 0].to_string())
    print(f"Wines with at least one |z| > 3: {(Z.abs() > 3).any(axis=1).sum()}")


def plot_boxplots(X, standardized, filename):
    data = X.to_numpy(dtype=float)
    if standardized:
        data = (data - data.mean(axis=0)) / data.std(axis=0, ddof=1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot(data)
    ax.set_xticks(range(1, X.shape[1] + 1))
    ax.set_xticklabels(X.columns, rotation=45, ha="right")
    if standardized:
        ax.set_ylabel("standardized value (z)")
        ax.set_title("Wine variables after standardization")
        ax.axhline(3, linestyle="--", linewidth=0.8)
        ax.axhline(-3, linestyle="--", linewidth=0.8)
    else:
        ax.set_ylabel("raw value")
        ax.set_title("Wine variables on their original scales")
    fig.tight_layout()

    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / filename, dpi=200)
    plt.close(fig)
    print(f"Saved figures/{filename}")


if __name__ == "__main__":
    X, y = load_wine()
    describe(X, y)
    print()
    plot_boxplots(X, standardized=False, filename="wine_boxplot_raw.png")
    plot_boxplots(X, standardized=True, filename="wine_boxplot_standardized.png")