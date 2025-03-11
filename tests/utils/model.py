import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

# Load the dataset
df = pd.read_csv("data/throttle.csv")

# Extract features and target variable
T = df["Throttle"].values  # Throttle
v = df["Speed"].values  # Speed
a = df["Acceleration"].values  # Acceleration

def accel_model(v, T, c1, k, c2):
    return (c1 * T) / (1 + k * v) - c2 * v**2

def fit_function(X, c1, k, c2):
    T, v = X  # Unpack the input variables
    return (c1 * T) / (1 + k * v) - c2 * v**2

# Initial guess for c1, k, and c2
initial_guess = [1.0, 0.1, 0.01]

# Fit the model
params, covariance = curve_fit(fit_function, (T, v), a, p0=initial_guess)

# Extract the fitted parameters
c1_opt, k_opt, c2_opt = params
print(f"Optimized Parameters: c1 = {c1_opt}, k = {k_opt}, c2 = {c2_opt}")

a_pred = fit_function((T, v), c1_opt, k_opt, c2_opt)

# Compute R² score
from sklearn.metrics import r2_score
r2 = r2_score(a, a_pred)
print("R² Score:", r2)

T_new, v_new = 1, 30  # Example throttle and speed
a_new = fit_function((T_new, v_new), c1_opt, k_opt, c2_opt)
print("Predicted Acceleration:", a_new)

