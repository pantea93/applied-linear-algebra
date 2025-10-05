import numpy as np


# ------------------------------------------------------------
# Function to check if a given square matrix is orthogonal
# ------------------------------------------------------------
def is_orthogonal(Q, tol=1e-8):
    I = np.eye(Q.shape[0])             # Identity matrix of same size
    return np.allclose(Q.T @ Q, I, atol=tol)   # Check if QᵀQ ≈ I

# ------------------------------------------------------------
# Get matrix input from the user
# ------------------------------------------------------------
n = int(input("Enter the dimension of the square matrix: "))

print("Enter the entries of the matrix row by row (space separated):")
A = np.zeros((n, n))
for i in range(n):
    row = list(map(float, input(f"Row {i+1}: ").split()))
    A[i, :] = row

print("\nMatrix A:\n", A)

# ------------------------------------------------------------
# Check orthogonality
# ------------------------------------------------------------
print("\nAᵀA =\n", A.T @ A)
print("\nIs A orthogonal? ->", is_orthogonal(A))

# ------------------------------------------------------------
# Compute determinant
# ------------------------------------------------------------
det_A = np.linalg.det(A)
print("\nDeterminant of A =", det_A)

# ------------------------------------------------------------
# QR Factorization
# ------------------------------------------------------------
Q, R = np.linalg.qr(A)
print("\nQ =\n", Q)
print("R =\n", R)

# Verify A = QR
print("\nQR =\n", Q @ R)
print("Is A = QR ? ->", np.allclose(A, Q @ R))

# ------------------------------------------------------------
# Check orthogonality of Q obtained from QR decomposition
# ------------------------------------------------------------
print("\nIs Q from QR decomposition orthogonal? ->", is_orthogonal(Q))

