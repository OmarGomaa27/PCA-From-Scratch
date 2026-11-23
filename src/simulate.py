import numpy as np


def simulate_pca(n, seed=0, lam_true=(9.0, 4.0, 1.0)):
    """Return X (n x p), the true directions Q_true (p x p) and lam_true (p).

    lam_true can be changed for the stress test, e.g. (9, 4, 4).
    """
    lam_true = np.asarray(lam_true, dtype=float)
    p = len(lam_true)
    rng = np.random.default_rng(seed)
    Q_true, _ = np.linalg.qr(rng.normal(size=(p, p)))      # true directions
    Sigma_true = Q_true @ np.diag(lam_true) @ Q_true.T      # true covariance
    X = rng.multivariate_normal(np.zeros(p), Sigma_true, size=n)
    return X, Q_true, lam_true


if __name__ == "__main__":
    from methods import my_pca_power

    X, Q_true, lam_true = simulate_pca(n=500, seed=0)
    lam, W = my_pca_power(X, k=3)
    print("X shape:", X.shape)
    print("my eigenvalues:", lam.round(3), "  (Iteration 2: 8.653 3.670 1.045)")
    print("true          :", lam_true)