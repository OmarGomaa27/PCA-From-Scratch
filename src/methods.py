import numpy as np


# Step 1 (Definition 2): sample mean vector
#   x̄ = (1/n) X^T 1            X: n x p  ->  x̄: p
def my_mean(X):
    n = X.shape[0]
    ones = np.ones(n)
    return (X.T @ ones) / n


# Step 2 (Definition 2): centred data matrix
#   X_c = X - 1 x̄^T            X: n x p,  x̄: p  ->  X_c: n x p
def my_centre(X, xbar):
    n = X.shape[0]
    ones = np.ones(n)
    return X - np.outer(ones, xbar)


# Step 3 (Definition 13): standardization
#   s_j = sqrt( (1/(n-1)) * sum_i (X_c)_ij^2 ),   D = diag(s_1, ..., s_p)
#   Z = X_c D^{-1}             X_c: n x p  ->  Z: n x p,  s: p
def my_standardize(Xc):
    n = Xc.shape[0]
    ones = np.ones(n)
    s = np.sqrt((ones @ (Xc * Xc)) / (n - 1))
    Z = Xc @ np.diag(1 / s)
    return Z, s


# Step 4 (Definition 3): sample covariance matrix
#   C = (1/(n-1)) X_c^T X_c      X_c: n x p  ->  C: p x p
def my_covariance(Xc):
    n = Xc.shape[0]
    return (Xc.T @ Xc) / (n - 1)


# Step 5a: power iteration -- top eigenpair of a symmetric PSD matrix A
#   w_{t+1} = A w_t / ||A w_t||,   ||u|| = sqrt(u^T u)   (Definition 1)
#   stop when ||w_{t+1} - w_t|| < tol;  then  lambda = w^T A w   (Theorem 4(b))
def my_power_iteration(A, tol=1e-12, max_iter=100000, seed=0):
    p = A.shape[0]
    rng = np.random.default_rng(seed)
    w = rng.normal(size=p)                 # random start
    w = w / np.sqrt(w @ w)                 # make it a unit vector
    for t in range(1, max_iter + 1):
        Aw = A @ w
        norm = np.sqrt(Aw @ Aw)
        if norm == 0:                      # A w = 0: remaining eigenvalue is 0
            return 0.0, w, t
        w_new = Aw / norm
        diff = np.sqrt((w_new - w) @ (w_new - w))
        w = w_new
        if diff < tol:                     # w has stopped changing
            break
    lam = w @ A @ w                        # Rayleigh quotient
    return lam, w, t


# Step 5b (Theorem 10, spectral theorem): deflation
#   A' = A - lambda w w^T   removes the found eigenpair, leaves the others
def my_deflate(A, lam, w):
    return A - lam * np.outer(w, w)


# Step 5c: first k eigenpairs by power iteration + deflation
#   lambda_1 >= ... >= lambda_k,   W_k = [w_1 ... w_k]   (p x k)
def my_eigen(C, k):
    p = C.shape[0]
    lam = np.zeros(k)
    W = np.zeros((p, k))
    iters = np.zeros(k, dtype=int)
    A = C.copy()
    for i in range(k):
        lam[i], W[:, i], iters[i] = my_power_iteration(A, seed=i)
        A = my_deflate(A, lam[i], W[:, i])
    return lam, W, iters


# Step 6: sort eigenpairs so that lambda_1 >= ... >= lambda_k
#   (Theorem 4: w_1 is then the direction of maximum variance)
def my_sort(lam, W):
    order = np.argsort(-lam)
    return lam[order], W[:, order]


# Step 7 (Definition 14, Proposition 9): explained-variance ratios
#   r_i = lambda_i / tr(C),   rho_k = r_1 + ... + r_k
#   tr(C) = sum of ALL p eigenvalues (Theorem 8), so this is correct
#   even when only k < p eigenvalues were computed
def my_explained_variance(lam, total):
    r = lam / total
    rho = np.cumsum(r)
    return r, rho


# Step 7 (cont.): smallest k with rho_k >= threshold (we use 0.90)
def my_choose_k(rho, threshold=0.90):
    hits = np.nonzero(rho >= threshold)[0]
    return int(hits[0]) + 1 if len(hits) > 0 else None


# Step 8: principal component scores
#   T = X_c W_k                 X_c: n x p,  W_k: p x k  ->  T: n x k
def my_scores(Xc, Wk):
    return Xc @ Wk


# Route A, Steps 1-6 in one call: returns lambda_1..lambda_k and W_k
def my_pca_power(X, k, standardize=False):
    xbar = my_mean(X)
    Xc = my_centre(X, xbar)
    if standardize:
        Xc, _ = my_standardize(Xc)
    C = my_covariance(Xc)
    lam, W, _ = my_eigen(C, k)
    return my_sort(lam, W)


# Route B (Theorem 7): thin SVD from scratch, first k terms
#   X_c^T X_c = V Sigma^2 V^T   ->  eigenpairs give v_i and sigma_i^2
#   u_i = X_c v_i / sigma_i     (requires sigma_i > 0)
def my_svd(Xc, k):
    lam, V, _ = my_eigen(Xc.T @ Xc, k)
    lam, V = my_sort(lam, V)
    sigma = np.sqrt(np.maximum(lam, 0.0))   # clip tiny negative rounding
    U = (Xc @ V) / sigma
    return U, sigma, V


# Helper: flip column signs of W to match W_ref (eigenvectors are
# defined only up to sign, Theorem 7(c))
def my_align_signs(W, W_ref):
    sgn = np.sign(np.sum(W * W_ref, axis=0))
    sgn[sgn == 0] = 1
    return W * sgn


# Verification helper (checking only, so numpy.linalg is allowed)
def report(name, mine, ref):
    """Print max |mine - ref| and the relative error ||mine - ref|| / ||ref||."""
    mine, ref = np.asarray(mine, float), np.asarray(ref, float)
    abs_err = np.abs(mine - ref).max()
    rel_err = np.linalg.norm(mine - ref) / np.linalg.norm(ref)
    print(f"{name:32s} max|diff| = {abs_err:.2e}   rel = {rel_err:.2e}")


# -------------------------------------------------------------------
if __name__ == "__main__":
    X = np.array([[1.0, 2.0],
                  [3.0, 4.0],
                  [5.0, 0.0],
                  [7.0, 2.0]])
    n = X.shape[0]

    # Step 1
    xbar = my_mean(X)
    print("my_mean:", xbar)
    print("check  :", X.mean(axis=0))

    # Step 2
    Xc = my_centre(X, xbar)
    print("my_centre:\n", Xc)
    print("check    :\n", X - X.mean(axis=0))
    print("column sums (should be 0):", np.ones(n) @ Xc)

    # Step 3
    Z, s = my_standardize(Xc)
    print("s:", s)
    print("variance of each Z column (should be 1):",
          np.ones(n) @ (Z * Z) / (n - 1))

    # Step 4
    C = my_covariance(Xc)
    print("C:\n", C)
    print("check:\n", np.cov(X, rowvar=False))
    print("symmetric (Theorem 1(a)):", np.allclose(C, C.T))
    print("diagonal = s^2:", np.diag(C), s**2)

    # Step 5
    lam, W, iters = my_eigen(C, k=2)
    lam_lib, W_lib = np.linalg.eigh(C)              # library: check only
    print("my eigenvalues :", lam, " iterations:", iters)
    print("eigh (sorted)  :", lam_lib[::-1])
    print("max |C w - lam w| :", np.abs(C @ W - W * lam).max())
    print("W^T W (should be I):\n", W.T @ W)
    print("sum lam vs tr(C) (Theorem 8):", lam.sum(), np.trace(C))

    # Steps 6-8
    lam, W = my_sort(lam, W)
    r, rho = my_explained_variance(lam, np.trace(C))
    print("ratios r:", r, " cumulative rho:", rho)
    print("chosen k (rho >= 0.90):", my_choose_k(rho))
    T = my_scores(Xc, W[:, :1])
    print("scores on PC1:", T.ravel())
    print("variance of scores = lambda_1 (Theorem 4):",
          (T.ravel() @ T.ravel()) / (n - 1), lam[0])

    # Wrapper and Route B, compared with the library
    lam_w, W_w = my_pca_power(X, k=2)
    U, sigma, V = my_svd(Xc, k=2)
    U_lib, s_lib, Vt_lib = np.linalg.svd(Xc, full_matrices=False)  # check only
    report("my_pca_power eigenvalues", lam_w, lam_lib[::-1])
    report("my_pca_power W (aligned)", my_align_signs(W_w, W_lib[:, ::-1]), W_lib[:, ::-1])
    report("my_svd sigma", sigma, s_lib)
    report("my_svd V (aligned)", my_align_signs(V, Vt_lib.T), Vt_lib.T)
    report("lambda = sigma^2/(n-1) (Thm 7)", sigma**2 / (n - 1), lam_w)