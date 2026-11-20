import numpy as np

# Step 1 (Definition 2): sample mean vector

#   x̄ = (1/n) X^T 1            X: n x p  ->  x̄: p
def my_mean(X):
    n = X.shape[0]
    ones = np.ones(n)
    return (X.T @ ones) / n


# -------------------------------------------------------------------
if __name__ == "__main__":
    X = np.array([[1.0, 2.0],
                  [3.0, 4.0],
                  [5.0, 0.0],
                  [7.0, 2.0]])
    print("my_mean:", my_mean(X))
    print("check  :", X.mean(axis=0))