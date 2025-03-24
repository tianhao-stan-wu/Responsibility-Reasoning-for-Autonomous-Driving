import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Load your data
csv_filename = "data/pwm2.csv"  # Change this to your actual file
data = pd.read_csv(csv_filename)
ax = data["acc_x"].values  # Longitudinal acceleration
vx = data["v_x"].values  # Velocity
pwm = data["pwm"].values  # Control input
m = 1845   # Example mass of the vehicle in kg (replace with actual mass)

# Define the cost function (sum of squared errors)
def cost_function(params):
    Cm1, Cm2, Cr0, Cr2 = params
    predicted_ax = ((Cm1 - Cm2 * vx) * pwm - Cr0 - Cr2 * vx**2) / m
    return np.sum((ax - predicted_ax) ** 2)  # Least squares error

# Set constraints to enforce positive parameters
bounds = [(0, None), (0, None), (0, None), (0, None)]  # All parameters >= 0

# Initial guesses (small positive values to start)
initial_guess = [5000, 50, 1000, 0.5]

# Optimize
result = minimize(cost_function, initial_guess, bounds=bounds, method="L-BFGS-B")

# Extract optimized parameters
Cm1_opt, Cm2_opt, Cr0_opt, Cr2_opt = result.x

print(f"Optimized Parameters:")
print(f"Cm1 = {Cm1_opt}")
print(f"Cm2 = {Cm2_opt}")
print(f"Cr0 = {Cr0_opt}")
print(f"Cr2 = {Cr2_opt}")


def compute_pwm(ax, vx, Cm1, Cm2, Cr0, Cr2, m):
    numerator = ax * m + Cr0 + Cr2 * vx**2
    denominator = Cm1 - Cm2 * vx
    if denominator == 0:
        print(f"Warning: Division by zero at vx = {vx}")
        return None
    pwm_value = numerator / denominator
    print(f"vx = {vx:.2f}, ax = {ax:.2f} -> pwm = {pwm_value:.4f}")
    return pwm_value


test_cases = [(0.5, 10), (-1.0, 5), (2.0, 20), (5, 0), (5, 30), (-5, 30)]  # (ax, vx) pairs

print("\nComputed pwm values for given (vx, ax):")
for ax_test, vx_test in test_cases:
    compute_pwm(ax_test, vx_test, Cm1_opt, Cm2_opt, Cr0_opt, Cr2_opt, m)