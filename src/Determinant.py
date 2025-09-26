import numpy as np

# ---- Example usage ----
n = int(input("Enter matrix size (n for n×n): "))
print("Enter the matrix row by row, with spaces between numbers:")
rows = []
for i in range(n):
    row = list(map(float, input(f"Row {i+1}: ").split()))
    rows.append(row)

A = np.array(rows)
print("Matrix A:")
print(A)

# Use NumPy's built-in determinant function
detA = np.linalg.det(A)
print("Determinant of A =", detA)

