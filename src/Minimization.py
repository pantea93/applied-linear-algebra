from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
# Create 30 evenly spaced values between 0 and 10
x = np.linspace(0, 10, 30).reshape(-1, 1)

# Generate corresponding y values with some random noise
# True relationship: y = 4x + 2 + noise
y = 4 * x.flatten() + 2 + np.random.randn(30) * 2

# Create and train the Linear Regression model
# The model automatically finds the best-fit line that minimizes squared error
model = LinearRegression()
model.fit(x, y)

# Predict the output (y values) using the trained model
y_pred = model.predict(x)

# Plot the data points and the fitted regression line
plt.scatter(x, y, label='Data')                 # Actual data points
plt.plot(x, y_pred, color='red', label='Fitted Line')  # Best-fit line
plt.title("Automatic Minimization using Scikit-learn")
plt.legend()
plt.show()

# Display the learned parameters
print("θ0 =", model.intercept_)    # Intercept term (bias)
print("θ1 =", model.coef_[0])      # Slope (weight for x)
