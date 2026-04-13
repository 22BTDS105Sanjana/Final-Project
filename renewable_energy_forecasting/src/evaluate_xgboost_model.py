import os
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from data_preprocessing import load_and_preprocess_data

# ================= LOAD DATA =================
df, X, y, features = load_and_preprocess_data()

# ================= SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# ================= LOAD MODEL =================
model_path = os.path.join(
    os.getcwd(),
    "renewable_energy_forecasting",
    "models",
    "solar_xgb_model.pkl"
)

print("Model path:", model_path)

if not os.path.exists(model_path):
    print("Model not found. Train first.")
    exit()

model = joblib.load(model_path)

print("Model loaded")

# ================= PREDICT =================
pred = model.predict(X_test)

# ================= METRICS =================
mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

print("\nXGBoost Performance:")
print("MAE :", mae)
print("RMSE:", rmse)
print("R2  :", r2)

# ================= PLOT =================
plt.figure(figsize=(12, 6))

plt.plot(y_test.values[:200], label="Actual", linewidth=2)
plt.plot(pred[:200], label="Predicted", linestyle="--")

plt.title("Solar Power Prediction (XGBoost)")
plt.xlabel("Time")
plt.ylabel("AC Power")

plt.legend()
plt.grid()

plt.show()