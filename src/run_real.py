"""
run_real.py -- PCA on the real Wine data and verification (Iteration 3).

Cleaning: none needed. All 178 x 13 entries are observed (Iteration 2),
no rows are removed, and the cultivar label is kept out of X.

1. Raw vs standardized PCA with our own functions (Q4).
2. Library check: scikit-learn PCA and the correlation matrix, to 1e-8.
3. Q4 demo: multiply one variable by 1000 and watch C and PC1 change.
4. Q5: scree plot, explained variance, choice of k.
5. PC1-PC2 scatter coloured by cultivar.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA          # check only

from load_data import load_wine
from methods import my_mean, my_centre, my_standardize, my_covariance, \
    my_eigen, my_sort, my_explained_variance, my_choose_k, my_scores, \
    my_pca_power, my_align_signs, report

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"


def pca_all_steps(X, standardize):
    """Steps 1-8 one by one; returns everything the report needs."""
    xbar = my_mean(X)                                  # Step 1
    Xc = my_centre(X, xbar)                            # Step 2
    if standardize:
        Xc, s = my_standardize(Xc)                     # Step 3
    C = my_covariance(Xc)                              # Step 4
    lam, W, iters = my_eigen(C, k=C.shape[0])          # Step 5
    lam, W = my_sort(lam, W)                           # Step 6
    r, rho = my_explained_variance(lam, np.trace(C))   # Step 7
    T = my_scores(Xc, W[:, :2])                        # Step 8 (first 2 PCs)
    return dict(Xc=Xc, C=C, lam=lam, W=W, iters=iters, r=r, rho=rho, T=T)


def show_loadings(W, names, k=2, top=4):
    for i in range(k):
        order = np.argsort(-np.abs(W[:, i]))[:top]
        parts = [f"{names[j]} {W[j, i]:+.3f}" for j in order]
        print(f"  PC{i+1} largest loadings: " + ", ".join(parts))


if __name__ == "__main__":
    Xdf, y = load_wine()
    names = list(Xdf.columns)
    X = Xdf.to_numpy(dtype=float)
    n, p = X.shape
    print(f"Data: n = {n}, p = {p}; rows removed: 0; missing values: "
          f"{int(np.isnan(X).sum())}\n")

    # ---------------------------------------------------------------- 1
    print("=" * 70)
    print("1. Raw vs standardized PCA (our functions)")
    print("=" * 70)
    raw = pca_all_steps(X, standardize=False)
    std = pca_all_steps(X, standardize=True)
    for tag, res in [("RAW (covariance C)", raw), ("STANDARDIZED (correlation R)", std)]:
        print(tag)
        print("  eigenvalues   :", res["lam"].round(4))
        print("  ratios r_i    :", res["r"].round(4))
        print("  cumulative rho:", res["rho"].round(4))
        print("  power-iteration steps per PC:", res["iters"])
        show_loadings(res["W"], names)
    print("  tr(R) =", round(np.trace(std["C"]), 10), "(should be p = 13)\n")

    # ---------------------------------------------------------------- 2
    print("=" * 70)
    print("2. Library check (scikit-learn PCA, numpy correlation matrix)")
    print("=" * 70)
    # raw: sklearn centres the data itself
    lib = PCA(n_components=p).fit(X)
    report("raw eigenvalues vs sklearn", raw["lam"], lib.explained_variance_)
    report("raw components vs sklearn",
           my_align_signs(raw["W"], lib.components_.T), lib.components_.T)
    # standardized: give sklearn the same Z (divisor n-1, Definition 13)
    Z = my_standardize(my_centre(X, my_mean(X)))[0]
    lib_z = PCA(n_components=p).fit(Z)
    report("std eigenvalues vs sklearn", std["lam"], lib_z.explained_variance_)
    report("std components vs sklearn",
           my_align_signs(std["W"], lib_z.components_.T), lib_z.components_.T)
    # independent route: eigenvalues of numpy's correlation matrix
    lam_R = np.linalg.eigvalsh(np.corrcoef(X, rowvar=False))[::-1]
    report("std eigenvalues vs corrcoef", std["lam"], lam_R)
    report("R vs numpy corrcoef", std["C"], np.corrcoef(X, rowvar=False))
    print()

    # ---------------------------------------------------------------- 3
    print("=" * 70)
    print("3. Q4: multiply one variable (alcohol) by 1000")
    print("=" * 70)
    j = names.index("alcohol")
    X1000 = X.copy()
    X1000[:, j] *= 1000
    C0 = raw["C"]
    C1 = my_covariance(my_centre(X1000, my_mean(X1000)))
    print(f"  C[alcohol, alcohol]: {C0[j, j]:.4f} -> {C1[j, j]:.1f}   "
          f"(ratio {C1[j, j] / C0[j, j]:.0f} = 1000^2)")
    k = names.index("proline")
    print(f"  C[alcohol, proline]: {C0[j, k]:.4f} -> {C1[j, k]:.1f}   "
          f"(ratio {C1[j, k] / C0[j, k]:.0f} = 1000)")
    lam_a, W_a = my_pca_power(X, k=1)
    lam_b, W_b = my_pca_power(X1000, k=1)
    print(f"  raw PC1 before: largest loading {names[np.argmax(np.abs(W_a[:, 0]))]}"
          f" ({np.abs(W_a[:, 0]).max():.4f}), lambda_1 = {lam_a[0]:.1f}")
    print(f"  raw PC1 after : largest loading {names[np.argmax(np.abs(W_b[:, 0]))]}"
          f" ({np.abs(W_b[:, 0]).max():.4f}), lambda_1 = {lam_b[0]:.1f}")
    lam_s, W_s = my_pca_power(X, k=2, standardize=True)
    lam_t, W_t = my_pca_power(X1000, k=2, standardize=True)
    report("standardized PCA: lambda unchanged", lam_t, lam_s)
    report("standardized PCA: W unchanged",
           my_align_signs(W_t, W_s), W_s)
    print()

    # ---------------------------------------------------------------- 4
    print("=" * 70)
    print("4. Q5: scree plot and choice of k (standardized data)")
    print("=" * 70)
    lam, rho = std["lam"], std["rho"]
    k90 = my_choose_k(rho, 0.90)
    k80 = my_choose_k(rho, 0.80)
    kaiser = int(np.sum(lam > 1))
    print(f"  smallest k with rho_k >= 0.90: {k90} (rho = {rho[k90-1]:.4f})")
    print(f"  smallest k with rho_k >= 0.80: {k80} (rho = {rho[k80-1]:.4f})")
    print(f"  eigenvalues > 1 (above the average 1 = tr(R)/p): {kaiser}")
    print(f"  first two PCs explain rho_2 = {rho[1]:.4f}")

    FIG_DIR.mkdir(exist_ok=True)
    idx = np.arange(1, p + 1)
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(idx, lam, "o-", label=r"eigenvalue $\lambda_i$")
    ax1.axhline(1, linestyle=":", color="gray", label=r"$\lambda = 1$ (average)")
    ax1.set_xlabel("component $i$")
    ax1.set_ylabel(r"$\lambda_i$")
    ax1.set_xticks(idx)
    ax2 = ax1.twinx()
    ax2.plot(idx, rho, "s--", color="tab:red", label=r"cumulative $\rho_k$")
    ax2.axhline(0.90, linestyle="--", color="tab:red", linewidth=0.8)
    ax2.set_ylabel(r"cumulative $\rho_k$")
    ax2.set_ylim(0, 1.05)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="center right")
    ax1.set_title("Scree plot, standardized Wine data")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "wine_scree.png", dpi=200)
    plt.close(fig)
    print("  Saved figures/wine_scree.png\n")

    # ---------------------------------------------------------------- 5
    print("=" * 70)
    print("5. PC1-PC2 scores coloured by cultivar")
    print("=" * 70)
    T = std["T"]
    fig, ax = plt.subplots(figsize=(6, 5))
    for c, m in zip([1, 2, 3], ["o", "s", "^"]):
        sel = (y.to_numpy() == c)
        ax.scatter(T[sel, 0], T[sel, 1], marker=m, alpha=0.7, label=f"cultivar {c}")
    ax.set_xlabel(f"PC1 ({std['r'][0]:.1%} of variance)")
    ax.set_ylabel(f"PC2 ({std['r'][1]:.1%} of variance)")
    ax.set_title("Wine: scores on the first two PCs")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "wine_pc1_pc2.png", dpi=200)
    plt.close(fig)
    for c in [1, 2, 3]:
        sel = (y.to_numpy() == c)
        print(f"  cultivar {c}: mean PC1 = {T[sel, 0].mean():+.3f}, "
              f"mean PC2 = {T[sel, 1].mean():+.3f}")
    print("  Saved figures/wine_pc1_pc2.png")