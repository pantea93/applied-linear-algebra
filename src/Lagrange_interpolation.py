import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Sample points and function values
# ---------------------------------------------------------
# t_points are the distinct interpolation points
t_points = np.array([0, 0.5, 1])
# y_points are the corresponding values of the function y = e^t
y_points = np.exp(t_points)

# ---------------------------------------------------------
# 2. Define function for Lagrange basis polynomial L_k(t)
# ---------------------------------------------------------
def lagrange_basis(t, t_points, k):
    """
    Compute the k-th Lagrange basis polynomial L_k(t)
    L_k(t) = Π_{i ≠ k} (t - t_i) / (t_k - t_i)
    """
    L = 1
    n = len(t_points)
    for i in range(n):
        if i != k:
            L *= (t - t_points[i]) / (t_points[k] - t_points[i])
    return L

# ---------------------------------------------------------
# 3. Define function for the full Lagrange polynomial p(t)
# ---------------------------------------------------------
def lagrange_polynomial(t, t_points, y_points):
    """
    Compute the Lagrange interpolating polynomial:
    p(t) = Σ y_k * L_k(t) for k = 0..n
    """
    p = 0
    n = len(t_points)
    for k in range(n):
        p += y_points[k] * lagrange_basis(t, t_points, k)
    return p

# ---------------------------------------------------------
# 4. Evaluate the polynomial for smooth plotting
# ---------------------------------------------------------
t_fine = np.linspace(0, 1, 100)  # 100 points between 0 and 1
# Evaluate the Lagrange polynomial at each point in t_fine
y_poly = np.array([lagrange_polynomial(t, t_points, y_points) for t in t_fine])
# Evaluate the exact function for comparison
y_exact = np.exp(t_fine)

# ---------------------------------------------------------
# 5. Plot the results
# ---------------------------------------------------------
plt.plot(t_fine, y_poly, 'r-', label='Lagrange Polynomial p(t)')  # Interpolating polynomial
plt.plot(t_fine, y_exact, 'b--', label='Exact e^t')              # Exact function
plt.scatter(t_points, y_points, color='black', zorder=5, label='Interpolation Points')  # Sample points
plt.xlabel('t')
plt.ylabel('y')
plt.title('Quadratic Lagrange Interpolation of e^t')
plt.legend()
plt.grid(True)
plt.show()
# ---------------------------------------------------------
# Step 6: Print polynomial values at the data points to verify interpolation
# ---------------------------------------------------------
for i, t in enumerate(t_points):
    print(f"p({t}) = {lagrange_interpolation(t):.5f}  (true e^{t} = {np.exp(t):.5f})")

