import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import load_model

from data_preprocessing import load_and_preprocess_data


# ================= LOAD DATA =================
df, X, y, features = load_and_preprocess_data()

# ================= LOAD SCALER =================
model_dir = os.path.join(os.getcwd(), "renewable_energy_forecasting", "models")
scaler_path = os.path.join(model_dir, "lstm_scaler.pkl")

scaler = joblib.load(scaler_path)

# ================= SCALE =================
X_scaled = scaler.transform(X)

# ================= RESHAPE =================
X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# ================= SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, shuffle=False
)

# ================= LOAD MODEL =================
model_path = os.path.join(model_dir, "solar_lstm_model.h5")

model = load_model(model_path, compile=False)

print("LSTM model loaded")

# ================= PREDICT =================
pred = model.predict(X_test).flatten()

# ================= METRICS =================
mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

print("\nLSTM Performance:")
print("MAE :", mae)
print("RMSE:", rmse)
print("R2  :", r2)

# ================= SMOOTH VISUAL =================
actual_series = pd.Series(y_test.values)
pred_series = pd.Series(pred)

smooth_actual = actual_series.rolling(window=5).mean()
smooth_pred = pred_series.rolling(window=5).mean()

# ================= PLOT =================
plt.figure(figsize=(12, 6))

plt.plot(smooth_actual[:200], label="Actual", linewidth=2)
plt.plot(smooth_pred[:200], label="Predicted", linestyle="--")

plt.title("Solar Power Prediction (LSTM - Smoothed)")
plt.xlabel("Time Steps")
plt.ylabel("AC Power")

plt.legend()
plt.grid()

plt.show()