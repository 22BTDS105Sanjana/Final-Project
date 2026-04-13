import numpy as np
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

# ================= LOAD DATA =================
from data_preprocessing import load_and_preprocess_data

df, X, y, features = load_and_preprocess_data()

# reduce size
X = X.iloc[:15000]
y = y.iloc[:15000]

# ================= SPLIT =================
split = int(0.8 * len(X))

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

# ================= LOAD MODEL =================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

model = joblib.load(os.path.join(MODEL_DIR, "solar_xgb_model.pkl"))

# ================= CALIBRATION =================
cal_size = int(0.2 * len(X_train))

X_cal = X_train.iloc[-cal_size:]
y_cal = y_train.iloc[-cal_size:]

y_cal_pred = model.predict(X_cal)

errors = np.abs(y_cal - y_cal_pred)

# ================= CONFIDENCE LEVEL =================
alpha = 0.1   # 90% confidence
q = np.quantile(errors, 1 - alpha)

# ================= TEST PREDICTIONS =================
y_pred = model.predict(X_test)

lower = y_pred - q
upper = y_pred + q

# ================= PRINT RESULTS =================
print("\n===== CONFORMAL PREDICTION RESULTS =====")

for i in range(5):
    print(f"\nSample {i}")
    print(f"Prediction: {y_pred[i]:.2f}")
    print(f"Interval: [{lower[i]:.2f}, {upper[i]:.2f}]")

# ================= 🔥 FIXED PLOT =================

# Take smaller slice (zoom view)
n = 100

y_true_plot = y_test.values[:n]
y_pred_plot = y_pred[:n]
lower_plot = lower[:n]
upper_plot = upper[:n]

# 🔥 Optional: force visibility if interval too small
min_width = 5
lower_plot = np.minimum(lower_plot, y_pred_plot - min_width)
upper_plot = np.maximum(upper_plot, y_pred_plot + min_width)

plt.figure(figsize=(12, 6))

# Main lines
plt.plot(y_true_plot, label="Actual", linewidth=2)
plt.plot(y_pred_plot, label="Prediction", linewidth=2)

# 🔥 Confidence interval (VISIBLE)
plt.fill_between(
    np.arange(n),
    lower_plot,
    upper_plot,
    alpha=0.4,
    label="Confidence Interval"
)

# 🔥 Boundary lines (IMPORTANT FIX)
plt.plot(lower_plot, linestyle='dashed', linewidth=1)
plt.plot(upper_plot, linestyle='dashed', linewidth=1)

plt.legend()
plt.title("Prediction with Confidence Interval (FIXED & VISIBLE)")
plt.xlabel("Time")
plt.ylabel("AC Power")

plt.grid(True)
plt.tight_layout()
plt.show()