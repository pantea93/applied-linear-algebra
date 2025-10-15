import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Sample data points (t_i, y_i)
# ---------------------------------------------------------
# These represent measured values, for example time (t) and observations (y)
t = np.array([0, 1, 2, 3, 4, 5])        # Independent variable values
y = np.array([2.2, 2.8, 3.6, 4.5, 5.1, 5.9])  # Dependent variable (measurements)

# ---------------------------------------------------------
# 2. Build the matrix A and vector y
# ---------------------------------------------------------
# Model: y = α + βt  →  A = [1, t_i]
# Each row of A represents one data point: [1, t_i]
A = np.column_stack((np.ones(len(t)), t))  # Create matrix A with a column of ones and t values
y = y.reshape(-1, 1)                       # Convert y to a column vector

# ---------------------------------------------------------
# 3. Solve the normal equation for least squares
#    x* = (A^T A)^(-1) A^T y
# ---------------------------------------------------------
# This gives the best-fit parameters [α, β]
x_star = np.linalg.inv(A.T @ A) @ A.T @ y

alpha = x_star[0, 0]  # Intercept (α)
beta = x_star[1, 0]   # Slope (β)

print("α (intercept):", round(alpha, 4))
print("β (slope):", round(beta, 4))

# ---------------------------------------------------------
# 4. Compute the least squares error
# ---------------------------------------------------------
# Error = ||Ax* - y||^2
y_pred = A @ x_star                      # Predicted values
error = np.linalg.norm(y_pred - y)**2    # Squared Euclidean norm
print("Least Squares Error:", round(error, 4))

# ---------------------------------------------------------
# 5. Plot the data and the best-fit line
# ---------------------------------------------------------
plt.scatter(t, y, color='blue', label='Data points')  # Plot the original measurements
plt.plot(t, y_pred, color='red', label='Best-fit line')  # Plot the fitted line
plt.xlabel('t')
plt.ylabel('y')
plt.title('Linear Data Fitting using Least Squares')
plt.legend()
plt.grid(True)
plt.show()

