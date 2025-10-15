import numpy as np
import matplotlib.pyplot as plt

def polynomial_interpolation(t_points, y_points, t_eval=None):
    """
    Interpolate a polynomial of degree n at given points.
    
    Parameters:
        t_points : array-like
            The interpolation points (x-values).
        y_points : array-like
            Function values at the interpolation points (y-values).
        t_eval : array-like, optional
            Points where the interpolating polynomial should be evaluated.
            If None, t_points will be used.
    
    Returns:
        coefficients : array
            Polynomial coefficients [α0, α1, ..., αn].
        y_eval : array
            Polynomial evaluated at t_eval points.
    """
    t_points = np.array(t_points)
    y_points = np.array(y_points)
    
    degree = len(t_points) - 1  # maximum degree of interpolating polynomial
    A = np.vander(t_points, N=degree+1, increasing=True)  # Vandermonde matrix
    
    # Solve for coefficients
    coefficients = np.linalg.solve(A, y_points)
    
    # Evaluate polynomial
    if t_eval is None:
        t_eval = t_points
    t_eval = np.array(t_eval)
    y_eval = sum(coeff * t_eval**i for i, coeff in enumerate(coefficients))
    
    return coefficients, y_eval

# ---------------------------------------------------------
# Example usage: Interpolating e^t at three points
# ---------------------------------------------------------
t_points = [0, 0.5, 1]
y_points = np.exp(t_points)
t_fine = np.linspace(0, 1, 100)

coeffs, y_poly = polynomial_interpolation(t_points, y_points, t_eval=t_fine)

print("Polynomial coefficients (α0, α1, ..., αn):")
print(coeffs)

# Plot
plt.plot(t_fine, y_poly, 'r-', label='Interpolating Polynomial')
plt.plot(t_fine, np.exp(t_fine), 'b--', label='Exact e^t')
plt.scatter(t_points, y_points, color='black', zorder=5, label='Interpolation points')
plt.xlabel('t')
plt.ylabel('y')
plt.title('Polynomial Interpolation of e^t')
plt.legend()
plt.grid(True)
plt.show()

