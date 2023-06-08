import numpy as np

def gramschmidt(A):
    Q = np.zeros(np.shape(A))
    for i in range(np.shape(A)[1]):
        Q[:, i] = A[:, i]
        for j in range(i):
            Q[:, i] -= (np.dot(Q[:, j], A[:, i]) / np.dot(Q[:, j], Q[:, j])) * Q[:, j]

    print(Q)
    for i in range(np.shape(A)[1]):
        Q[:, i] = Q[:, i] / np.linalg.norm(Q[:, i])

    return Q

N = 1000
K = 3

# A = np.random.random((N, K))
# A = np.loadtxt("hw_gramschmidt.csv", delimiter=',', dtype=float)
A = np.array([[4, 2, 1], [2, 4, 2], [1, 2, 4]])
Q = gramschmidt(A)
print(Q)

## Rank Check
print(f"Rank = {np.linalg.matrix_rank(np.concatenate((A, Q), axis=1))}")

## Q*Q Check
print(f"Identity Error <= {np.max(np.abs(np.identity(K) - Q.T @ Q))}")

