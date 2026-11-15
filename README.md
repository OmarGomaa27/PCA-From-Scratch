# PCA from Scratch: What Drives Variation in Real Data?

DS5020 Intro Linear Algebra & Probability — Final Project, Fall 2026

Omar Gomaa

- Report: [`docs/PCA_From_Scratch.pdf`](docs/PCA_From_Scratch.pdf)
- Overleaf (read-only): <https://www.overleaf.com/project/6af784b96729c38d6b66ca0e/share#ceda48968ce5ee71718743159a721c8fa10d77f0c626dac4>
- Project plan: [`docs/Project Plan.xlsx`](docs/Project%20Plan.xlsx)

## Project

Principal Component Analysis derived and implemented from scratch: the covariance matrix
and its spectral decomposition, the Lagrange-multiplier derivation of the maximum-variance
direction, the equivalence of the eigendecomposition and SVD routes, and the
explained-variance ratios. Iteration 2 covers the mathematics, the data description and a
simulation with known truth; Iteration 3 applies the method to the Wine data with an
eigensolver written by hand (power iteration with deflation).

## Data

UCI Wine dataset (ID 109): <https://archive.ics.uci.edu/dataset/109/wine>.
178 wines × 13 chemical measurements, no missing values. One row is one wine, one column is
one measurement. The cultivar label (1–3) is not part of X and is used only to colour plots.

The raw file is small (10 KB) and is committed as `data/wine.data`, with the original UCI
documentation in `data/wine.names`. To re-download it:

```bash
curl -o data/wine.data https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data
```

The dataset page above also offers the same data as a zip
(`https://archive.ics.uci.edu/static/public/109/wine.zip`), which contains `wine.data` and
`wine.names`. Both URLs were verified on 14 November 2026.

## Structure

```
src/        data loading and simulation scripts
data/       raw data (wine.data, wine.names, Index)
figures/    generated plots
docs/       report PDF, project plan, script output logs
```

## Requirements

Python 3.11+ with `numpy`, `pandas` and `matplotlib`:

```bash
pip install numpy pandas matplotlib
```

## Running

From the project root:

```bash
python src/load_data.py > docs/data_summary.txt
```

Prints the data summary (size, types, missing values, summary statistics, variance shares,
outlier counts) and writes
`figures/wine_boxplot_raw.png` and `figures/wine_boxplot_standardized.png`.

```bash
python src/simulation.py > docs/simulation_output.txt
```

Runs the known-truth simulation at n = 50, 500 and 5000, checking each derivation
numerically, then repeats 200 times at each of seven sample sizes to show the error shrinking
with n. The plot is
`figures/sim_error_vs_n.png`. All randomness uses `np.random.default_rng(0)`, so the numbers
reproduce exactly.

## Simulation with known truth

Data are drawn from N(0, Σ) in p = 3 dimensions with Σ = Q diag(9, 4, 1) Qᵀ for a random
orthogonal Q, so the true principal directions and variances are known. The run confirms that
C is symmetric and positive semi-definite, that the variance along w₁ equals λ₁, that
WᵀW = I, that the SVD and eigendecomposition agree, and that tr(C) = Σλᵢ — each to machine
precision. The mean relative eigenvalue error follows the theoretical 2/√(πn) curve, with a
fitted log–log slope of −0.500.
