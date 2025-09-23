import numpy as np

def ldv_factorization(A):
    """
    Compute the LDV factorization of a square matrix A.
    A = L * D * V
    L: lower unitriangular matrix
    D: diagonal matrix
    V: upper unitriangular matrix
    """
    A = A.astype(float)  # ensure floating point operations
    n = A.shape[0]
    
    # Step 1: Perform Gaussian elimination to obtain U
    # and collect multipliers to form L
    U = A.copy()
    L = np.eye(n)  # start L as the identity matrix
    for i in range(n - 1):
        for j in range(i + 1, n):
            if U[i, i] == 0:
                raise ValueError("Zero pivot encountered!")  # no division by zero
            m = U[j, i] / U[i, i]  # multiplier for elimination
            L[j, i] = m            # store multiplier in L
            U[j, :] -= m * U[i, :] # eliminate entry below the pivot
    
    # Step 2: Extract the diagonal matrix D from U
    D = np.zeros((n, n))
    for i in range(n):
        D[i, i] = U[i, i]
    
    # Step 3: Normalize U by dividing each row by its diagonal element to get V
    V = np.zeros((n, n))
    for i in range(n):
        V[i, :] = U[i, :] / D[i, i]
    
    return L, D, V

# ---- Take matrix from user ----
n = int(input("Enter the size of the matrix n (e.g. 3 for 3x3): "))
A = np.zeros((n, n))
print("Enter the matrix row by row:")
for i in range(n):
    row = input(f"Row {i+1} (space separated numbers): ").split()
    A[i, :] = [float(x) for x in row]

# Perform LDV factorization
L, D, V = ldv_factorization(A)

# Print results
print("\nL =")
print(L)
print("\nD =")
print(D)
print("\nV =")
print(V)

# Check by reconstructing A
A_reconstructed = L @ D @ V
print("\nReconstructed A =")
print(A_reconstructed)
