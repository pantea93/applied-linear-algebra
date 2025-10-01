import numpy as np

# Read the vector x from user
x_input = input("Enter the components of vector x (space separated): ").split()
x = np.array([float(val) for val in x_input])

# Read the new basis matrix A
print("\nEnter the new basis matrix (each row in a new line, space separated):")
rows = []
for i in range(len(x)):  # number of rows = dimension of vector x
    row = input(f"Row {i+1}: ").split()
    rows.append([float(num) for num in row])
A = np.array(rows)

# Solve the system A * c = x  → coordinates of x in new basis
c = np.linalg.solve(A, x)

# Output
print("\nCoordinates of x in the new basis:", c)
