"""
check_simulation.py -- Simulation test: recover the known truth (Iteration 3).

1. Library check (n = 500): my_pca_power vs scikit-learn PCA, to 1e-8.
2. Truth check (n = 50, 500, 5000): each lambda_hat_i within 3 standard
   errors 3 * lambda_i * sqrt(2/n); directions match Q_true up to sign.
3. Error vs n: mean error over 200 seeds at 7 sample sizes, log-log plot.
4. Stress test: true variances (9, 4, 4) -- PC2 and PC3 are not well defined.

The PCA itself uses only our own functions from methods.py.
scikit-learn and numpy.linalg appear only as checks.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA          # check only

from methods import my_pca_power, my_mean, my_centre, my_covariance, \
    my_eigen, my_sort, my_align_signs, report
from simulate import simulate_pca

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"


def angle_deg(a, b):
    """Angle between two directions, ignoring sign (degrees)."""
    c = abs(a @ b) / np.sqrt((a @ a) * (b @ b))
    return np.degrees(np.arccos(min(c, 1.0)))


# ------------------------------------------------------------------ 1
def library_check(n=500):
    print("=" * 70)
    print(f"1. Library check (n = {n}, seed = 0)")
    print("=" * 70)
    X, Q_true, lam_true = simulate_pca(n=n, seed=0)
    lam, W = my_pca_power(X, k=3)

    p = PCA(n_components=3).fit(X)
    W_lib = p.components_.T
    report("eigenvalues vs sklearn", lam, p.explained_variance_)
    report("components vs sklearn", my_align_signs(W, W_lib), W_lib)
    report("W^T W vs I (Corollary 6)", W.T @ W, np.eye(3))
    print()


# ------------------------------------------------------------------ 2
def truth_check(sizes=(50, 500, 5000)):
    print("=" * 70)
    print("2. Truth check: lambda_hat within 3 SE = 3 lambda sqrt(2/n)?")
    print("=" * 70)
    for n in sizes:
        X, Q_true, lam_true = simulate_pca(n=n, seed=0)
        lam, W = my_pca_power(X, k=3)
        se3 = 3 * lam_true * np.sqrt(2 / n)
        print(f"n = {n}")
        print(f"  {'i':>2} {'true':>6} {'est':>8} {'|diff|':>8} {'3 SE':>8} "
              f"{'pass':>5} {'angle':>8}")
        for i in range(3):
            d = abs(lam[i] - lam_true[i])
            print(f"  {i+1:>2} {lam_true[i]:>6.1f} {lam[i]:>8.3f} {d:>8.3f} "
                  f"{se3[i]:>8.3f} {str(d < se3[i]):>5} "
                  f"{angle_deg(W[:, i], Q_true[:, i]):>7.2f}d")
        print("  |W^T Q_true| (identity up to sign):")
        print("  " + str(np.abs(W.T @ Q_true).round(3)).replace("\n", "\n  "))
    print()


# ------------------------------------------------------------------ 3
def error_vs_n(sizes=(50, 100, 200, 500, 1000, 2000, 5000), reps=200):
    print("=" * 70)
    print(f"3. Error vs n (mean over seeds 0..{reps-1})")
    print("=" * 70)
    lam_true = np.array([9.0, 4.0, 1.0])
    mean_err = np.zeros((len(sizes), 3))
    mean_ang = np.zeros(len(sizes))
    for a, n in enumerate(sizes):
        errs, angs = [], []
        for seed in range(reps):
            X, Q_true, _ = simulate_pca(n=n, seed=seed)
            lam, W = my_pca_power(X, k=3)
            errs.append(np.abs(lam - lam_true) / lam_true)
            angs.append(angle_deg(W[:, 0], Q_true[:, 0]))
        mean_err[a] = np.mean(errs, axis=0)
        mean_ang[a] = np.mean(angs)

    print(f"{'n':>6} {'rel err l1':>11} {'rel err l2':>11} {'rel err l3':>11} "
          f"{'theory':>8} {'angle w1':>9}")
    for a, n in enumerate(sizes):
        print(f"{n:>6} {mean_err[a,0]:>11.4f} {mean_err[a,1]:>11.4f} "
              f"{mean_err[a,2]:>11.4f} {2/np.sqrt(np.pi*n):>8.4f} "
              f"{mean_ang[a]:>8.2f}d")
    slope = np.polyfit(np.log(sizes), np.log(mean_err.mean(axis=1)), 1)[0]
    print(f"Fitted log-log slope: {slope:.3f}   (theory: -0.5)")

    sizes = np.array(sizes)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for i in range(3):
        ax.loglog(sizes, mean_err[:, i], "o-",
                  label=fr"$\hat\lambda_{i+1}$ (true {lam_true[i]:g})")
    ax.loglog(sizes, 2 / np.sqrt(np.pi * sizes), "k--",
              label=r"theory $2/\sqrt{\pi n}$")
    ax.set_xlabel("sample size $n$")
    ax.set_ylabel(r"mean relative error of $\hat\lambda_i$")
    ax.set_title(f"From-scratch PCA: error vs $n$ ({reps} seeds)")
    ax.legend()
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "check_sim_error_vs_n.png", dpi=200)
    plt.close(fig)
    print("Saved figures/check_sim_error_vs_n.png\n")


# ------------------------------------------------------------------ 4
def stress_test(n=500, seeds=range(5)):
    print("=" * 70)
    print(f"4. Stress test: lam_true = (9, 4, 4), n = {n}  (Assumption 7 broken)")
    print("=" * 70)
    print(f"{'seed':>4} {'lam_hat':>24} {'iters PC2/PC3':>14} "
          f"{'angle w2':>9} {'angle w3':>9} {'subspace err':>13}")
    for seed in seeds:
        for lam_true, tag in [((9.0, 4.0, 1.0), "normal"),
                              ((9.0, 4.0, 4.0), "equal ")]:
            X, Q_true, _ = simulate_pca(n=n, seed=seed, lam_true=lam_true)
            Xc = my_centre(X, my_mean(X))
            lam, W, iters = my_eigen(my_covariance(Xc), k=3)
            lam, W = my_sort(lam, W)
            # span{w2, w3} vs span{q2, q3}: compare projection matrices
            P_hat = W[:, 1:] @ W[:, 1:].T
            P_true = Q_true[:, 1:] @ Q_true[:, 1:].T
            sub_err = np.abs(P_hat - P_true).max()
            print(f"{seed:>4} {tag} {str(lam.round(3)):>17} "
                  f"{iters[1]:>6}/{iters[2]:<6} "
                  f"{angle_deg(W[:, 1], Q_true[:, 1]):>8.1f}d "
                  f"{angle_deg(W[:, 2], Q_true[:, 2]):>8.1f}d "
                  f"{sub_err:>12.3f}")
    print("With equal true variances, the angles of w2 and w3 to the true axes\n"
          "are large and change from seed to seed, and power iteration needs\n"
          "more steps (ratio lam3/lam2 near 1). The PLANE span{w2, w3} is still\n"
          "recovered: only the individual directions inside it are undefined.\n")


if __name__ == "__main__":
    library_check()
    truth_check()
    error_vs_n()
    stress_test()