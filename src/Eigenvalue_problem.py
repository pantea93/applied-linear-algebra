import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm

# --- Step 1: Input matrix A ---
n = int(input("Enter the size of the square matrix A (e.g. 2 or 3): "))
A = np.zeros((n, n))
print("Enter each row of matrix A (numbers separated by spaces):")
for i in range(n):
    row = input(f"Row {i+1}: ").strip().split()
    A[i, :] = [float(x) for x in row]

print("\nMatrix A:")
print(A)

# --- Step 2: Eigen decomposition ---
eigvals, eigvecs = np.linalg.eig(A)
print("\nEigenvalues:")
print(eigvals)
print("\nEigenvectors:")
print(eigvecs)

# --- Step 3: Stability check ---
print("\n--- Stability Analysis ---")
for i, val in enumerate(eigvals):
    if np.real(val) < 0:
        print(f"λ{i+1} = {val:.4f} → Stable (decays over time)")
    elif np.real(val) > 0:
        print(f"λ{i+1} = {val:.4f} → Unstable (grows over time)")
    else:
        print(f"λ{i+1} = {val:.4f} → Neutral (oscillates or constant)")

# --- Step 4: Simulate time evolution ---
t_values = np.linspace(0, 2, 100)  # time from 0 to 2 seconds
u0 = np.ones((n, 1))               # initial condition u(0) = [1, 1, ...]^T

solutions = []
for t in t_values:
    u_t = expm(A * t) @ u0
    solutions.append(u_t.flatten())

solutions = np.array(solutions)

# --- Step 5: Plot ---
plt.figure(figsize=(8, 5))
for i in range(n):
    plt.plot(t_values, solutions[:, i], label=f'u{i+1}(t)')
plt.xlabel("Time t")
plt.ylabel("u(t)")
plt.title("Time Evolution of the System du/dt = Au")
plt.legend()
plt.grid(True)
plt.show()

