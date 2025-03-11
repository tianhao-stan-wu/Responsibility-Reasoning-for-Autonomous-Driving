import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# # Load the dataset
# df = pd.read_csv("data/throttle.csv")

# # Extract features and target variable
# T = df["Throttle"].values.reshape(-1, 1)  # Throttle (reshaped for sklearn)
# a = df["Acceleration"].values  # Acceleration

# # Fit the linear regression model
# model = LinearRegression()
# model.fit(T, a)

# # Extract fitted parameters (slope and intercept)
# slope = model.coef_[0]
# intercept = model.intercept_
# print(f"Fitted Linear Model: a = {slope:.4f} * T + {intercept:.4f}")

# # Predictions
# a_pred = model.predict(T)

# # Compute R² score
# r2 = r2_score(a, a_pred)
# print("R² Score:", r2)

# x = 0.5
# # Predict acceleration for a new throttle value
# T_new = np.array([[x]])  # Example throttle value
# a_new = model.predict(T_new)
# print("Predicted Acceleration for T=",x, ":", a_new[0])

def throttle_model(a=None, T=None):

    # model is a = 11.5 * T - 5.6

    if a is not None:
        T = (a + 5.6) / 11.5
        return T

    elif T is not None:
        a = 11.5 * T - 5.6
        return a

print(throttle_model(T=1))
print(throttle_model(T=0))
print(throttle_model(a=0))