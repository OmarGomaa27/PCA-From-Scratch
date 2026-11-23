"""
Small worked example, step by step (Iteration 3).

Four points in R^2, chosen so that every quantity is an exact fraction:
    (13, 23), (21, 17), (1, 7), (9, 1)

Part A follows the Iteration 2 algorithm by hand with fractions.Fraction
(Q1: centre, C, characteristic polynomial, eigenvectors, projection).
Part B computes the same components from the SVD of X_c (Q2).
Part C runs our from-scratch functions in methods.py on the same points
and checks them against the exact answer.
"""
from fractions import Fraction as F

import numpy as np

from methods import my_pca_power, my_svd, my_mean, my_centre, my_scores, \
    my_align_signs, report


# ---------- tiny exact helpers (lists of Fractions) ----------
def transpose(A):
    return [list(r) for r in zip(*A)]


def matmul(A, B):
    Bt = transpose(B)
    return [[sum(a * b for a, b in zip(row, col)) for col in Bt] for row in A]


def exact_sqrt(q):
    """Square root of a Fraction that is a perfect square (else error)."""
    num, den = q.numerator, q.denominator
    rn, rd = int(round(num ** 0.5)), int(round(den ** 0.5))
    assert rn * rn == num and rd * rd == den, f"{q} is not a perfect square"
    return F(rn, rd)


def show(name, M):
    if isinstance(M[0], list):
        print(f"{name} =")
        for row in M:
            print("   [" + ", ".join(f"{str(v):>7}" for v in row) + " ]")
    else:
        print(f"{name} = [" + ", ".join(str(v) for v in M) + "]")


POINTS = [[13, 23], [21, 17], [1, 7], [9, 1]]

if __name__ == "__main__":
    X = [[F(v) for v in row] for row in POINTS]
    n, p = len(X), len(X[0])
    print("=" * 64)
    print("Part A (Q1): step by step with exact fractions")
    print("=" * 64)
    show("X", X)

    # Step 1: x̄ = (1/n) X^T 1
    xbar = [sum(X[i][j] for i in range(n)) / n for j in range(p)]
    print("\nStep 1  xbar = (1/n) X^T 1")
    show("xbar", xbar)

    # Step 2: X_c = X - 1 x̄^T
    Xc = [[X[i][j] - xbar[j] for j in range(p)] for i in range(n)]
    print("\nStep 2  X_c = X - 1 xbar^T")
    show("X_c", Xc)
    show("column sums of X_c (Definition 2: must be 0)",
         [sum(Xc[i][j] for i in range(n)) for j in range(p)])

    # Step 4: C = (1/(n-1)) X_c^T X_c   (Step 3, standardizing, not used here)
    XtX = matmul(transpose(Xc), Xc)
    C = [[v / (n - 1) for v in row] for row in XtX]
    print("\nStep 4  C = (1/(n-1)) X_c^T X_c")
    show("X_c^T X_c", XtX)
    show("C", C)
    print("C symmetric (Theorem 1(a)):", C[0][1] == C[1][0])

    # Step 5: eigenvalues from the characteristic polynomial
    #   det(C - t I) = t^2 - tr(C) t + det(C) = 0
    tr = C[0][0] + C[1][1]
    det = C[0][0] * C[1][1] - C[0][1] * C[1][0]
    disc = tr * tr - 4 * det
    root = exact_sqrt(disc)
    lam1, lam2 = (tr + root) / 2, (tr - root) / 2
    print("\nStep 5  characteristic polynomial  t^2 - tr(C) t + det(C) = 0")
    print(f"   tr(C) = {tr},  det(C) = {det}")
    print(f"   t^2 - ({tr}) t + ({det}) = 0")
    print(f"   discriminant = tr^2 - 4 det = {disc},  sqrt = {root}")
    print(f"   lambda_1 = {lam1},  lambda_2 = {lam2}")

    # Eigenvectors: (C - lambda I) w = 0. From row 1:
    #   (c11 - lambda) w1 + c12 w2 = 0  ->  w ∝ (c12, lambda - c11)
    W = []
    for lam in (lam1, lam2):
        v = [C[0][1], lam - C[0][0]]
        norm = exact_sqrt(v[0] * v[0] + v[1] * v[1])
        w = [vi / norm for vi in v]
        W.append(w)
        print(f"   lambda = {lam}:  (C - lambda I) w = 0  ->  w prop. to {[str(x) for x in v]}"
              f",  ||.|| = {norm},  unit w = {[str(x) for x in w]}")
    w1, w2 = W

    # Step 6 is automatic (lambda_1 >= lambda_2 by construction).
    # Step 7: explained-variance ratios
    r1, r2 = lam1 / tr, lam2 / tr
    print(f"\nStep 7  r_1 = {r1},  r_2 = {r2}   (sum = {r1 + r2})")

    # Step 8: scores on PC1, T = X_c w_1
    T = [sum(Xc[i][j] * w1[j] for j in range(p)) for i in range(n)]
    print("\nStep 8  scores on PC1:  T = X_c w_1")
    show("T", T)
    varT = sum(t * t for t in T) / (n - 1)
    print(f"   variance of the scores = {varT}  (= lambda_1, Theorem 4)")

    # ---------- second-way checks (substitute back) ----------
    print("\nSecond-way checks (substitute the answer back):")
    for lam, w, name in [(lam1, w1, "1"), (lam2, w2, "2")]:
        Cw = [C[0][0] * w[0] + C[0][1] * w[1], C[1][0] * w[0] + C[1][1] * w[1]]
        lw = [lam * w[0], lam * w[1]]
        print(f"   C w_{name} = {[str(x) for x in Cw]},  lambda_{name} w_{name} = "
              f"{[str(x) for x in lw]},  equal: {Cw == lw}")
    print(f"   w_1 . w_2 = {w1[0]*w2[0] + w1[1]*w2[1]}  (Theorem 5)")
    print(f"   lambda_1 + lambda_2 = {lam1 + lam2} = tr(C) = {tr}  (Theorem 8)")

    # ---------- Part B: the SVD route ----------
    print("\n" + "=" * 64)
    print("Part B (Q2): the same components from the SVD of X_c")
    print("=" * 64)
    # X_c^T X_c = V Sigma^2 V^T, so its eigenvalues are sigma_i^2 = (n-1) lambda_i
    s1sq, s2sq = (n - 1) * lam1, (n - 1) * lam2
    s1, s2 = exact_sqrt(s1sq), exact_sqrt(s2sq)
    print(f"eigenvalues of X_c^T X_c: sigma_1^2 = {s1sq}, sigma_2^2 = {s2sq}")
    print(f"singular values: sigma_1 = {s1}, sigma_2 = {s2}")
    V = transpose([w1, w2])            # same eigenvectors as C (Theorem 7)
    u1 = [sum(Xc[i][j] * w1[j] for j in range(p)) / s1 for i in range(n)]
    u2 = [sum(Xc[i][j] * w2[j] for j in range(p)) / s2 for i in range(n)]
    show("u_1 = X_c v_1 / sigma_1", u1)
    show("u_2 = X_c v_2 / sigma_2", u2)
    U = transpose([u1, u2])
    show("U^T U (must be I)", matmul(transpose(U), U))
    Sigma = [[s1, F(0)], [F(0), s2]]
    rebuilt = matmul(matmul(U, Sigma), transpose(V))
    print("U Sigma V^T == X_c exactly:", rebuilt == Xc)
    print(f"lambda_i = sigma_i^2/(n-1): {s1sq/(n-1)}, {s2sq/(n-1)}  "
          f"(same as Part A: {lam1}, {lam2})")

    # ---------- Part C: our from-scratch functions ----------
    print("\n" + "=" * 64)
    print("Part C: methods.py on the same points (floating point)")
    print("=" * 64)
    Xf = np.array(POINTS, dtype=float)
    lam_exact = np.array([float(lam1), float(lam2)])
    W_exact = np.array([[float(w1[0]), float(w2[0])],
                        [float(w1[1]), float(w2[1])]])
    lam_my, W_my = my_pca_power(Xf, k=2)
    report("my_pca_power eigenvalues", lam_my, lam_exact)
    report("my_pca_power W (aligned)", my_align_signs(W_my, W_exact), W_exact)
    Xcf = my_centre(Xf, my_mean(Xf))
    U_my, s_my, V_my = my_svd(Xcf, k=2)
    report("my_svd sigma", s_my, [float(s1), float(s2)])
    T_my = my_scores(Xcf, my_align_signs(W_my, W_exact)[:, :1]).ravel()
    report("my_scores on PC1", T_my, [float(t) for t in T])