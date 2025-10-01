import sympy as sp

# --- User Input ---
# Get number of rows and columns
rows = int(input("Enter number of rows: "))
cols = int(input("Enter number of columns: "))

print("Enter the matrix entries row by row (space separated):")

# Build the matrix from user input
matrix_entries = []
for i in range(rows):
    row = list(map(int, input(f"Row {i+1}: ").split()))
    matrix_entries.append(row)

A = sp.Matrix(matrix_entries)

print("\nMatrix A:")
sp.pprint(A)

# --- Null Space (Kernel) ---
null_space = A.nullspace()
print("\n--- Null Space (Kernel) ---")
if null_space:
    print("Basis for Null Space:")
    for vec in null_space:
        sp.pprint(vec)
else:
    print("Only the zero vector (trivial solution)")

# --- Column Space ---
col_space = A.columnspace()
print("\n--- Column Space ---")
print("Basis for Column Space:")
for vec in col_space:
    sp.pprint(vec)

# --- Rank ---
rank = A.rank()
print("\nRank of A:", rank)

