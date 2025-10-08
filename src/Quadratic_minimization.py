import numpy as np
import matplotlib.pyplot as plt

# ---- Step 1: Define coefficients ----
# p(x, y) = 5x^2 + 3xy + 2y^2 - 4x + 6y - 7
K = np.array([[5, 3/2],
              [3/2, 2]])   # symmetric matrix
f = np.array([2, -3])       # vector from linear terms
c = -7                      # constant term

# ---- Step 2: Solve Kx = f to find minimizer ----
x_star = np.linalg.solve(K, f)
print("Minimizer (x*, y*):", x_star)

# ---- Step 3: Compute minimum value p(x*) = c - f^T K^{-1} f ----
p_min = c - f.T @ np.linalg.inv(K) @ f
print("Minimum value p(x*):", p_min)

# ---- Step 4: (Optional) visualize the quadratic surface ----
x_vals = np.linspace(-4, 4, 100)
y_vals = np.linspace(-4, 4, 100)
X, Y = np.meshgrid(x_vals, y_vals)

# Compute p(x, y) on the grid
P = 5*X**2 + 3*X*Y + 2*Y**2 - 4*X + 6*Y - 7

# ---- Step 5: Plot the surface ----
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, P, cmap='viridis', alpha=0.8)
ax.scatter(x_star[0], x_star[1], p_min, color='r', s=60, label='Minimum Point')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('p(x, y)')
ax.set_title('Quadratic Function Surface')
ax.legend()
plt.show()

