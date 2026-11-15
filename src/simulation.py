"""
simulation.py -- Simulation with known truth (Iteration 2, item 7).

Run from the project root:
    python src/simulation.py

Model: x ~ N(0, Sigma_true) in p = 3 dimensions, with
    Sigma_true = Q_true diag(9, 4, 1) Q_true^T,
so the true principal directions (columns of Q_true) and true variances
(9, 4, 1) are known. For n = 50, 500 and 5000 we run PCA two ways
(eigendecomposition of C, and SVD of X_c) and check every derivation:
    Q1  C symmetric, v^T C v = ||X_c v||^2 / (n-1) >= 0
    Q2  variance along w_1 equals lambda_1
    Q3  W^T W = I
    Q4  lambda_i = sigma_i^2 / (n-1), and V = W up to sign
    Q5  tr(C) = sum of lambda_i; explained-variance ratios
Then we repeat each n many times to show how the error shrinks with n.

Every sample is drawn with np.random.default_rng(0), as the brief requires.
For each n a fresh generator is created, so Q_true is identical for every n
and the n = 500 run reproduces the brief's starter code exactly.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "figures"

LAM_TRUE = np.array([9.0, 4.0, 1.0])   # true variances
P = len(LAM_TRUE)


def make_truth(rng):
    """Draw the true directions Q_true and build Sigma_true."""
    Q_true, _ = np.linalg.qr(rng.normal(size=(P, P)))
    Sigma_true = Q_true @ np.diag(LAM_TRUE) @ Q_true.T
    return Q_true, Sigma_true


def pca_eig(X):
    """Route A: centre, form C, eigendecompose, sort descending."""
    Xc = X - X.mean(axis=0)                       # Definition 2
    C = Xc.T @ Xc / (len(X) - 1)                  # Definition 3
    lam, W = np.linalg.eigh(C)                    # ascending order
    lam, W = lam[::-1], W[:, ::-1]                # sort descending
    return Xc, C, lam, W


def pca_svd(Xc):
    """Route B: thin SVD of the centred data."""
    _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    return s, Vt.T


def angle_deg(a, b):
    """Angle between two directions, ignoring sign (in degrees)."""
    c = abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


def single_run(n):
    """One run at sample size n: print the true-vs-estimate table and checks."""
    rng = np.random.default_rng(0)
    Q_true, Sigma_true = make_truth(rng)
    X = rng.multivariate_normal(np.zeros(P), Sigma_true, size=n)

    Xc, C, lam, W = pca_eig(X)
    s, V = pca_svd(Xc)

    print("=" * 64)
    print(f"n = {n}")
    print("=" * 64)

    # True vs estimate table
    print(f"{'i':>2} {'true lam':>9} {'est lam':>9} {'rel err':>8} "
          f"{'angle(w_i, q_i)':>16}")
    for i in range(P):
        rel = abs(lam[i] - LAM_TRUE[i]) / LAM_TRUE[i]
        print(f"{i+1:>2} {LAM_TRUE[i]:>9.3f} {lam[i]:>9.3f} {rel:>8.3%} "
              f"{angle_deg(W[:, i], Q_true[:, i]):>14.2f} deg")
    print()

    # Q1: symmetry and positive semi-definiteness
    v = rng.normal(size=P)
    lhs = v @ C @ v
    rhs = np.sum((Xc @ v) ** 2) / (n - 1)
    print("Q1  max |C - C^T|              :", f"{np.abs(C - C.T).max():.2e}")
    print("Q1  v^T C v  vs ||X_c v||^2/(n-1):", f"{lhs:.6f}  {rhs:.6f}")
    print("Q1  smallest eigenvalue >= 0    :", f"{lam.min():.4f}",
          lam.min() >= 0)

    # Q2: variance along w_1 equals lambda_1
    var_w1 = np.var(Xc @ W[:, 0], ddof=1)
    print("Q2  var(X_c w_1)  vs  lambda_1  :", f"{var_w1:.6f}  {lam[0]:.6f}")

    # Q3: orthonormal eigenvectors
    print("Q3  max |W^T W - I|            :",
          f"{np.abs(W.T @ W - np.eye(P)).max():.2e}")

    # Q4: SVD agrees with eigendecomposition
    lam_svd = s**2 / (n - 1)
    print("Q4  max |lambda - sigma^2/(n-1)|:",
          f"{np.abs(lam - lam_svd).max():.2e}")
    print("Q4  max ||V^T W| - I|           :",
          f"{np.abs(np.abs(V.T @ W) - np.eye(P)).max():.2e}")

    # Q5: total variance and explained-variance ratios
    ratios = lam / lam.sum()
    print("Q5  tr(C)  vs  sum(lambda)      :",
          f"{np.trace(C):.6f}  {lam.sum():.6f}")
    print("Q5  ratios (est)                :", np.round(ratios, 4))
    print("Q5  ratios (true)               :",
          np.round(LAM_TRUE / LAM_TRUE.sum(), 4))

    # Recovered directions vs truth (brief's check)
    print("|W^T Q_true| (identity up to sign):")
    print(np.abs(W.T @ Q_true).round(3))
    print()


def error_vs_n(sizes=(50, 100, 200, 500, 1000, 2000, 5000), reps=200):
    """Repeat each n many times; plot mean relative eigenvalue error vs n.

    For normal data with distinct eigenvalues, the large-sample standard
    deviation of lambda_hat_i / lambda_i is sqrt(2/n), so the mean absolute
    relative error is about sqrt(2/n) * sqrt(2/pi) = 2 / sqrt(pi n).
    That curve is drawn for reference: it has slope -1/2 on log-log axes.
    """
    rng = np.random.default_rng(0)
    Q_true, Sigma_true = make_truth(rng)

    mean_err = np.zeros((len(sizes), P))
    mean_ang = np.zeros((len(sizes), P))
    for a, n in enumerate(sizes):
        errs = np.zeros((reps, P))
        angs = np.zeros((reps, P))
        for r in range(reps):
            X = rng.multivariate_normal(np.zeros(P), Sigma_true, size=n)
            _, _, lam, W = pca_eig(X)
            errs[r] = np.abs(lam - LAM_TRUE) / LAM_TRUE
            angs[r] = [angle_deg(W[:, i], Q_true[:, i]) for i in range(P)]
        mean_err[a] = errs.mean(axis=0)
        mean_ang[a] = angs.mean(axis=0)

    print("=" * 64)
    print(f"Error vs n ({reps} repetitions per n)")
    print("=" * 64)
    print(f"{'n':>6} {'rel err l1':>11} {'rel err l2':>11} {'rel err l3':>11} "
          f"{'theory':>8} {'angle w1':>9}")
    for a, n in enumerate(sizes):
        theory = 2 / np.sqrt(np.pi * n)
        print(f"{n:>6} {mean_err[a,0]:>11.4f} {mean_err[a,1]:>11.4f} "
              f"{mean_err[a,2]:>11.4f} {theory:>8.4f} {mean_ang[a,0]:>8.2f}d")

    # Slope of log(error) against log(n), averaged over the 3 eigenvalues
    slope = np.polyfit(np.log(sizes), np.log(mean_err.mean(axis=1)), 1)[0]
    print(f"\nFitted log-log slope: {slope:.3f}  (theory: -0.5)")

    sizes = np.array(sizes)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for i in range(P):
        ax.loglog(sizes, mean_err[:, i], "o-",
                  label=fr"$\lambda_{i+1}$ (true {LAM_TRUE[i]:g})")
    ax.loglog(sizes, 2 / np.sqrt(np.pi * sizes), "k--",
              label=r"theory $2/\sqrt{\pi n}$")
    ax.set_xlabel("sample size $n$")
    ax.set_ylabel("mean relative error of $\\hat\\lambda_i$")
    ax.set_title(f"Eigenvalue error vs $n$ ({reps} repetitions)")
    ax.legend()
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "sim_error_vs_n.png", dpi=200)
    plt.close(fig)
    print("Saved figures/sim_error_vs_n.png")


if __name__ == "__main__":
    for n in (50, 500, 5000):
        single_run(n)
    error_vs_n()