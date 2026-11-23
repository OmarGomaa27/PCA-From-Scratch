"""
check_assumptions.py -- Check the Iteration 2 assumptions (Iteration 3).

Each assumption is checked first on the simulation (n = 500, seed = 0),
where it holds by construction, to show what "passing" looks like, and
then on the standardized Wine data. Numbering follows Iteration 2, Section 3.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from load_data import load_wine
from methods import my_mean, my_centre, my_standardize, my_covariance, \
    my_eigen, my_sort, my_scores
from simulate import simulate_pca

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"


def pca(X, standardize):
    Xc = my_centre(X, my_mean(X))
    s = np.sqrt(np.ones(len(X)) @ (Xc * Xc) / (len(X) - 1))
    if standardize:
        Xc, _ = my_standardize(Xc)
    C = my_covariance(Xc)
    lam, W, iters = my_eigen(C, k=C.shape[0])
    order = np.argsort(-lam)
    return Xc, s, C, lam[order], W[:, order], iters[order]


def angle_deg(a, b):
    c = abs(a @ b) / np.sqrt((a @ a) * (b @ b))
    return np.degrees(np.arccos(min(c, 1.0)))


def lag1_autocorr(t):
    t = t - t.mean()
    return (t[:-1] @ t[1:]) / (t @ t)


def linear_vs_quadratic_r2(x, y):
    """R^2 of a straight-line fit and of a quadratic fit of y on x."""
    out = []
    for deg in (1, 2):
        coef = np.polyfit(x, y, deg)
        resid = y - np.polyval(coef, x)
        out.append(1 - (resid @ resid) / ((y - y.mean()) @ (y - y.mean())))
    return out


def run_checks(name, X, standardize, groups=None, Q_true=None, names=None):
    n, p = X.shape
    if names is None:
        names = [f"x{j+1}" for j in range(p)]
    print("=" * 70)
    print(f"{name}: n = {n}, p = {p}")
    print("=" * 70)
    Xc, s, C, lam, W, iters = pca(X, standardize)
    T = my_scores(Xc, W[:, :2])

    # A1 independence: lag-1 autocorrelation of PC1 scores in file order
    if groups is None:
        ac = [lag1_autocorr(T[:, 0])]
    else:
        ac = [lag1_autocorr(T[groups == g, 0]) for g in np.unique(groups)]
    print(f"A1 independence: lag-1 autocorrelation of PC1 scores "
          f"{'(within each cultivar)' if groups is not None else ''}: "
          f"{np.round(ac, 3)}")
    sizes = [len(T)] if groups is None else [int((groups == g).sum()) for g in np.unique(groups)]
    print(f"     no-trend threshold 2/sqrt(group size): {np.round([2 / np.sqrt(m) for m in sizes], 3)}")

    # A2 complete data
    print(f"A2 complete data: missing values = {int(np.isnan(X).sum())}")

    # A3 n > p and full column rank
    print(f"A3 n > p: {n} > {p} = {n > p};  smallest eigenvalue = {lam[-1]:.4f} "
          f"> 0: {lam[-1] > 1e-10};  condition number lambda_1/lambda_p = "
          f"{lam[0] / lam[-1]:.1f}")

    # A4 linearity: most correlated pairs, linear vs quadratic R^2
    R = C if standardize else C / np.outer(s, s)
    iu = np.triu_indices(p, 1)
    top = np.argsort(-np.abs(R[iu]))[:3]
    print("A4 linearity: 3 most correlated pairs, R^2 linear vs quadratic fit")
    for t in top:
        a, b = iu[0][t], iu[1][t]
        r2l, r2q = linear_vs_quadratic_r2(X[:, a], X[:, b])
        print(f"     {names[a]} vs {names[b]}: r = {R[a, b]:+.3f}, R^2 lin = {r2l:.3f}, "
              f"quad = {r2q:.3f}, gain = {r2q - r2l:.3f}")

    # A5 variance = structure
    if Q_true is not None:
        print("A5 variance = structure: angle of w_i to the true direction: "
              + ", ".join(f"{angle_deg(W[:, i], Q_true[:, i]):.2f}d"
                          for i in range(p)))
    if groups is not None:
        # share of each PC's variance explained by the cultivar (eta^2)
        etas = []
        for i in range(2):
            t = T[:, i]
            between = sum((groups == g).sum() * (t[groups == g].mean() - t.mean()) ** 2
                          for g in np.unique(groups))
            etas.append(between / ((t - t.mean()) @ (t - t.mean())))
        print(f"A5 variance = structure: share of PC1 / PC2 variance explained "
              f"by cultivar (eta^2): {etas[0]:.3f} / {etas[1]:.3f}")

    # A6 comparable scales
    print(f"A6 scales: s_j from {s.min():.3f} to {s.max():.3f}, "
          f"ratio max/min = {s.max() / s.min():.1f};  all s_j > 0: {bool((s > 0).all())}")

    # A7 distinct eigenvalues: relative gaps and power-iteration steps
    gaps = (lam[:-1] - lam[1:]) / lam[:-1]
    print("A7 distinct eigenvalues: relative gap (lambda_i - lambda_i+1)/lambda_i")
    print("     gaps : " + " ".join(f"{g:.3f}" for g in gaps))
    print("     steps: " + " ".join(f"{int(k)}" for k in iters))
    print(f"     smallest gap: {gaps.min():.3f} between PC{np.argmin(gaps) + 1} "
          f"and PC{np.argmin(gaps) + 2}")

    # A8 outliers: |z| > 3, refit without them
    Z = Xc if standardize else Xc / s
    flagged = (np.abs(Z) > 3).any(axis=1)
    expected = n * p * 2 * 0.00135        # P(|N(0,1)| > 3) ~ 0.0027
    print(f"A8 outliers: entries with |z| > 3 = {int((np.abs(Z) > 3).sum())} "
          f"(about {expected:.1f} expected for normal data), rows flagged = "
          f"{int(flagged.sum())}")
    if flagged.any():
        _, _, _, lam2, W2, _ = pca(X[~flagged], standardize)
        print("     refit without flagged rows: angle change of PC1, PC2 = "
              f"{angle_deg(W[:, 0], W2[:, 0]):.2f}d, "
              f"{angle_deg(W[:, 1], W2[:, 1]):.2f}d;  lambda_1, lambda_2: "
              f"{lam[0]:.3f} -> {lam2[0]:.3f}, {lam[1]:.3f} -> {lam2[1]:.3f}")
    print()
    return T


if __name__ == "__main__":
    X_sim, Q_true, _ = simulate_pca(n=500, seed=0)
    run_checks("SIMULATION (assumptions hold by construction)", X_sim,
               standardize=False, Q_true=Q_true)

    Xdf, y = load_wine()
    X = Xdf.to_numpy(dtype=float)
    groups = y.to_numpy()
    T = run_checks("WINE (standardized)", X, standardize=True, groups=groups,
                   names=list(Xdf.columns))

    # Figure: PC1 scores against row index (A1)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for g, m in zip([1, 2, 3], ["o", "s", "^"]):
        idx = np.nonzero(groups == g)[0]
        ax.scatter(idx, T[idx, 0], marker=m, s=14, alpha=0.7, label=f"cultivar {g}")
    ax.set_xlabel("row index in file")
    ax.set_ylabel("PC1 score")
    ax.set_title("A1: PC1 scores in file order (no trend within cultivar)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "wine_scores_vs_index.png", dpi=200)
    plt.close(fig)
    print("Saved figures/wine_scores_vs_index.png")