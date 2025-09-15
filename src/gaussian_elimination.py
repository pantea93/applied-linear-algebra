import numpy as np
def gaussian_elimination(A, b):
    """
    Solves Ax = b using Gaussian elimination and back substitution.
    A : coefficient matrix (n x n)
    b : right-hand side vector (n)
    """
    n = len(b)
    # Augmented matrix [A|b]
    Ab = np.hstack([A.astype(float), b.reshape(-1,1).astype(float)])

    # Forward elimination (Gaussian elimination)
    for i in range(n):
        # Pivoting: make sure Ab[i,i] is not zero
        if Ab[i, i] == 0:
            for j in range(i+1, n):
                if Ab[j, i] != 0:
                    Ab[[i, j]] = Ab[[j, i]]  # swap rows
                    break
        
        # Make leading coefficient = 1
        Ab[i] = Ab[i] / Ab[i, i]
        
        # Eliminate below
        for j in range(i+1, n):
            Ab[j] = Ab[j] - Ab[j, i] * Ab[i]
    
    # Back substitution
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = Ab[i, -1] - np.dot(Ab[i, i+1:n], x[i+1:n])
    
    return x

# Get input from user
n = int(input("Number of equations (n): "))

print("Enter matrix A row by row (numbers separated by space):")
A = []
for i in range(n):
    row = list(map(float, input(f"Row {i+1}: ").split()))
    A.append(row)
A = np.array(A)

print("Enter vector b (numbers separated by space):")
b = np.array(list(map(float, input().split())))

solution = gaussian_elimination(A, b)
print("Solution of the system: ", solution)
