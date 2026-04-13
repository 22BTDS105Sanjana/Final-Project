import os
import numpy as np
import torch
import joblib

from sklearn.metrics import mean_absolute_error, mean_squared_error
from data_preprocessing import load_and_preprocess_data


# =========================
# LOAD DATA (FIXED)
# =========================

# Handles extra returned values safely
df,X, y,features= load_and_preprocess_data()

# Optional: limit for faster testing
X = X.iloc[:10000]
y = y.iloc[:10000]


# Base directory
base_dir = os.path.dirname(os.path.dirname(__file__))


# =========================
# RANDOM FOREST
# =========================

rf_path = os.path.join(base_dir, "models", "solar_model.pkl")
rf_model = joblib.load(rf_path)

rf_pred = rf_model.predict(X)

rf_mae = mean_absolute_error(y, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y, rf_pred))


# =========================
# XGBOOST
# =========================

xgb_path = os.path.join(base_dir, "models", "solar_xgb_model.pkl")
xgb_model = joblib.load(xgb_path)

xgb_pred = xgb_model.predict(X)

xgb_mae = mean_absolute_error(y, xgb_pred)
xgb_rmse = np.sqrt(mean_squared_error(y, xgb_pred))


# =========================
# LSTM (FIXED SCALING)
# =========================

from tensorflow.keras.models import load_model

lstm_model_path = os.path.join(base_dir, "models", "solar_lstm_model.h5")
lstm_model = load_model(lstm_model_path, compile=False)

# ✅ Load SAME scaler used during training
lstm_scaler_path = os.path.join(base_dir, "models", "lstm_scaler.pkl")
scaler = joblib.load(lstm_scaler_path)

X_scaled = scaler.transform(X)

# reshape for LSTM
X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

lstm_pred = lstm_model.predict(X_lstm)
lstm_pred = lstm_pred.flatten()

lstm_mae = mean_absolute_error(y, lstm_pred)
lstm_rmse = np.sqrt(mean_squared_error(y, lstm_pred))


# =========================
# TRANSFORMER (FIXED)
# =========================

from evaluate_transformer_model import TransformerModel

scaler_path = os.path.join(base_dir, "models", "transformer_scalers.pth")
scalers = torch.load(scaler_path, weights_only=False)

x_scaler = scalers["x_scaler"]
y_scaler = scalers["y_scaler"]

X_scaled = x_scaler.transform(X)

SEQ_LEN = 12


def create_sequences(X, seq_len):
    xs = []
    for i in range(len(X) - seq_len):
        xs.append(X[i:i + seq_len])
    return np.array(xs)


X_seq = create_sequences(X_scaled, SEQ_LEN)
X_tensor = torch.tensor(X_seq, dtype=torch.float32)

model_path = os.path.join(base_dir, "models", "solar_transformer_model.pth")

model = TransformerModel(X_tensor.shape[2])

# ✅ safer loading
model.load_state_dict(torch.load(model_path, weights_only=True))
model.eval()

with torch.no_grad():
    preds_scaled = model(X_tensor).numpy()

transformer_pred = y_scaler.inverse_transform(preds_scaled).flatten()

# Adjust y to match sequence length
y_transformer = y.iloc[SEQ_LEN:].values

trans_mae = mean_absolute_error(y_transformer, transformer_pred)
trans_rmse = np.sqrt(mean_squared_error(y_transformer, transformer_pred))


# =========================
# PRINT RESULTS
# =========================

print("\n========== Model Performance Comparison ==========\n")

print(f"Random Forest  - MAE: {rf_mae:.2f} | RMSE: {rf_rmse:.2f}")
print(f"XGBoost        - MAE: {xgb_mae:.2f} | RMSE: {xgb_rmse:.2f}")
print(f"LSTM           - MAE: {lstm_mae:.2f} | RMSE: {lstm_rmse:.2f}")
print(f"Transformer    - MAE: {trans_mae:.2f} | RMSE: {trans_rmse:.2f}")

print("\n=================================================\n")