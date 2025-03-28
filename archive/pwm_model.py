import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression



"""
Estimated Parameters:
Cm1 = 7870.5132
Cm2 = -56.8534
Cr0 = 1179.1085
Cr2 = -0.5040
"""


# Load CSV
csv_filename = "data/pwm2.csv"  # Change this to your actual file
data = pd.read_csv(csv_filename)

# Extract relevant data
ax = data["acc_x"].values  # Longitudinal acceleration
vx = data["v_x"].values  # Velocity
pwm = data["pwm"].values  # Control input
m = 1845  # Example mass (kg), replace with actual vehicle mass

# Prepare the feature matrix X and target y
X = np.column_stack([
    pwm,          # x1 = pwm
    -vx * pwm,    # x2 = -vx * pwm
    -np.ones_like(ax),  # x3 = -1
    -vx**2        # x4 = -vx^2
])

y = ax * m  # Target variable

# Fit linear regression model
model = LinearRegression(fit_intercept=False)  # No intercept since x3 accounts for it
model.fit(X, y)

# Extract estimated parameters
Cm1, Cm2, Cr0, Cr2 = model.coef_

Cm1 = 550*(3.45*0.919)/0.34
Cm1, Cm2, Cr0, Cr2 = (Cm1, 0, 50, 0.5)

# Print results
print(f"Estimated Parameters:")
print(f"Cm1 = {Cm1:.4f}")
print(f"Cm2 = {Cm2:.4f}")
print(f"Cr0 = {Cr0:.4f}")
print(f"Cr2 = {Cr2:.4f}")


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
    compute_pwm(ax_test, vx_test, Cm1, Cm2, Cr0, Cr2, m)



