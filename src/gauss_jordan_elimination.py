import numpy as np

# --- get matrices from user ---
n = int(input("Enter the size of square matrices (n): "))

print("Enter matrix A row by row (numbers separated by space):")
A = []
for i in range(n):
    row = list(map(float, input(f"Row {i+1}: ").split()))
    A.append(row)
A = np.array(A, dtype=float)

print("Enter matrix B row by row (numbers separated by space):")
B = []
for i in range(n):
    row = list(map(float, input(f"Row {i+1}: ").split()))
    B.append(row)
B = np.array(B, dtype=float)

# --- compute inverses with NumPy ---
try:
    A_inv = np.linalg.inv(A)
    B_inv = np.linalg.inv(B)
except np.linalg.LinAlgError:
    print("One of the matrices is not invertible!")
    exit()

# np.round(array, 6) rounds each element of 'array' to 6 decimal places.
# We use it here only for cleaner printing of floating-point results,
# so small numerical errors like 1.0000000002 become 1.0.
print("\nA^-1 =\n", np.round(A_inv, 6))
print("\nB^-1 =\n", np.round(B_inv, 6))

# --- check (AB)^-1 = B^-1 A^-1 ---
AB = A @ B
try:
    AB_inv = np.linalg.inv(AB)
except np.linalg.LinAlgError:
    print("(AB) is not invertible!")
    exit()

B_inv_A_inv = B_inv @ A_inv

print("\n(AB)^-1 =\n", np.round(AB_inv, 6))
print("\nB^-1 A^-1 =\n", np.round(B_inv_A_inv, 6))

# difference to show equality
print("\nDifference (should be near zero):\n", np.round(AB_inv - B_inv_A_inv, 6))
