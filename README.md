# PCA from Scratch: What Drives Variation in Real Data?

DS5020 Intro Linear Algebra & Probability — Final Project, Fall 2026

Omar Gomaa

Principal Component Analysis derived and implemented from scratch. The covariance matrix and
its spectral decomposition, the Lagrange-multiplier derivation of the maximum-variance
direction, the equivalence of the eigendecomposition and SVD routes and the explained-variance
ratios are proved in the report; the same results are then implemented as NumPy code, one
function per algorithm step, and checked three ways: against an exact four-point example in
rational arithmetic, against a simulation whose true components are known, and against
scikit-learn on the UCI Wine data.

Reports (Iteration 3 contains all of Iteration 2, plus the implementation and its verification):

| Iteration | PDF | Overleaf (read-only) |
| --- | --- | --- |
| 3 | [`docs/PCA_From_Scratch_i3.pdf`](docs/PCA_From_Scratch_i3.pdf) | <https://www.overleaf.com/project/6b02749a143915e4ba732f2a/share#f940cb5a558fa0c6e98990e613fb4392c2ccd5898fedc843> |
| 2 | [`docs/PCA_From_Scratch_i2.pdf`](docs/PCA_From_Scratch_i2.pdf) | <https://www.overleaf.com/project/6af784b96729c38d6b66ca0e/share#ceda48968ce5ee71718743159a721c8fa10d77f0c626dac4> |

Project plan: [`docs/Project Plan.xlsx`](docs/Project%20Plan.xlsx)

## Data

UCI Wine dataset (ID 109): <https://archive.ics.uci.edu/dataset/109/wine>.
178 wines × 13 chemical measurements, no missing values. One row is one wine, one column is one
measurement. The cultivar label (1–3) is not part of X and is used only to colour plots.

The raw file is small (10 KB) and is committed as `data/wine.data`, with the original UCI
documentation in `data/wine.names`. To re-download it:

```bash
curl -o data/wine.data https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data
```

## Requirements

Python 3 with `numpy`, `pandas` and `matplotlib`. `scikit-learn` is used only to verify results,
never to compute them:

```bash
pip install numpy pandas matplotlib scikit-learn
```

## Structure

```
src/        implementation and scripts
data/       raw data (wine.data, wine.names, Index)
figures/    generated plots
docs/       report PDFs, project plan, script output logs
```

`src/methods.py` holds the implementation: one `my_` function per numbered step of the algorithm
in the report, written with NumPy arrays and matrix products only, with the equation it computes
in a comment above it. Norms are computed as `sqrt(u @ u)`, eigenpairs by power iteration with
deflation. `numpy.linalg`, `numpy.corrcoef` and scikit-learn appear only in checking code. All
randomness goes through `np.random.default_rng(seed)` with seed 0, so every run reproduces the
numbers in the report and in `docs/*_output.txt`.

## Running

Run each script from the project root.

| Command | What it does | Produces |
| --- | --- | --- |
| `python src/load_data.py` | Iteration 2: describes the Wine data (size, types, missing values, summary statistics, variance shares, outliers) and draws the raw and standardized boxplots | `figures/wine_boxplot_raw.png`, `figures/wine_boxplot_standardized.png` |
| `python src/simulate.py` | The simulation model as a function, `simulate_pca(n, seed)`; run directly it is a quick reproduction check against the Iteration 2 estimates | — |
| `python src/small_example.py` | The four-point worked example, step by step in exact fractions, with the second-way checks and a floating-point comparison | — |
| `python src/check_simulation.py` | Simulation test: truth check at n = 50, 500, 5000, scikit-learn comparison, error against n, stress test with equal true variances | `figures/check_sim_error_vs_n.png` |
| `python src/run_real.py` | Wine data: raw versus standardized PCA, the ×1000 rescaling demonstration, scikit-learn comparison, scree plot and scores | `figures/wine_scree.png`, `figures/wine_pc1_pc2.png` |
| `python src/check_assumptions.py` | Checks every assumption on the simulation first, then on Wine | `figures/wine_scores_vs_index.png` |

Each script prints its results; the saved logs in `docs/` were produced by redirecting that
output:

```bash
python src/run_real.py > docs/run_real_output.txt
```

On Windows, PowerShell's `>` writes UTF-16 with a byte-order mark. To keep the logs as UTF-8,
either run the command from Git Bash as above, or in PowerShell use:

```bash
$env:PYTHONIOENCODING="utf-8"; python src/run_real.py | Out-File -Encoding utf8 docs/run_real_output.txt
```

The logs in `docs/` are `data_summary.txt`, `small_example_output.txt`,
`check_simulation_output.txt`, `run_real_output.txt` and `check_assumptions_output.txt`
(`simulation_output.txt` is the Iteration 2 log, kept for reference).
