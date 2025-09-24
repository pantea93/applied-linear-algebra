import numpy as np

def lu_decomposition(A):
    """
    Compute LU factorization without pivoting.
    Returns L and U such that A = L * U.
    """
    n = A.shape[0]
    L = np.eye(n)           # start with identity matrix for L
    U = A.copy().astype(float)  # make a copy of A for U

    for i in range(n):
        pivot = U[i, i]
        if pivot == 0:
            raise ValueError("Pivot is zero; row swapping (pivoting) needed.")

        for j in range(i+1, n):
            # factor to eliminate below the pivot
            factor = U[j, i] / pivot
            L[j, i] = factor            # store factor in L
            # subtract factor * pivot row from current row
            U[j, i:] = U[j, i:] - factor * U[i, i:]
    return L, U

def forward_substitution(L, b):
    """Solve Lc = b using forward substitution."""
    n = L.shape[0]
    c = np.zeros(n)
    for i in range(n):
        # compute c[i] using previously computed c[0:i]
        c[i] = b[i] - np.dot(L[i, :i], c[:i])
    return c

def back_substitution(U, c):
    """Solve Ux = c using back substitution."""
    n = U.shape[0]
    x = np.zeros(n)
    for i in reversed(range(n)):
        # compute x[i] from right to left using known x[i+1:]
        x[i] = (c[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
    return x

# ---- main program ----
n = int(input("Enter number of rows (n): "))
A = []
print("Enter matrix A row by row (numbers separated by space):")
for i in range(n):
    row = list(map(float, input(f"Row {i+1}: ").split()))
    A.append(row)
A = np.array(A)

b = np.array(list(map(float, input("Enter vector b (numbers separated by space): ").split())))

# compute LU
L, U = lu_decomposition(A)

# solve step by step
c = forward_substitution(L, b)
x = back_substitution(U, c)

print("\nMatrix L:")
print(L)
print("\nMatrix U:")
print(U)
print("\nSolution vector x:")
print(x)

