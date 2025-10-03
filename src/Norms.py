import sympy as sp

# --------------------------
# Get matrix from user
# --------------------------
n = int(input("Enter matrix size n (for n x n): "))
A_entries = []
for i in range(n):
    row = input(f"Row {i+1} (space separated): ").split()
    A_entries.append([sp.sympify(val) for val in row])

A = sp.Matrix(A_entries)

# --------------------------
# Matrix norms
# --------------------------

# 1-norm: max column sum
norm1 = max([sum(abs(A[i, j]) for i in range(n)) for j in range(n)])

# ∞-norm: max row sum
norm_inf = max([sum(abs(A[i, j]) for j in range(n)) for i in range(n)])


# --------------------------
# Results
# --------------------------
print("\nMatrix A:")
sp.pprint(A)

print("\nMatrix Norms:")
print(f"||A||_1   = {norm1}")
print(f"||A||_∞   = {norm_inf}")

