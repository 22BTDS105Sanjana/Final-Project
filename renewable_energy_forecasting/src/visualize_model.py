import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import torch.serialization

# ================= FIX SCALER LOAD =================
torch.serialization.add_safe_globals([MinMaxScaler])

# ================= PATHS =================
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

print("Model directory:", MODEL_DIR)
print("Available model files:", os.listdir(MODEL_DIR))

# ================= LOAD DATA =================
from data_preprocessing import load_and_preprocess_data

df, X, y, features = load_and_preprocess_data()

# Reduce size
X = X.iloc[:15000]
y = y.iloc[:15000]

# ================= SPLIT =================
split = int(0.8 * len(X))

X_test = X.iloc[split:]
y_test = y.iloc[split:]

# ================= LOAD MODELS =================
rf_model = joblib.load(os.path.join(MODEL_DIR, "solar_model.pkl"))
xgb_model = joblib.load(os.path.join(MODEL_DIR, "solar_xgb_model.pkl"))

lstm_model = load_model(
    os.path.join(MODEL_DIR, "solar_lstm_model.h5"),
    compile=False
)
lstm_scaler = joblib.load(os.path.join(MODEL_DIR, "lstm_scaler.pkl"))

transformer_scalers = torch.load(
    os.path.join(MODEL_DIR, "transformer_scalers.pth"),
    weights_only=False
)

# ================= TRANSFORMER MODEL =================
class TransformerModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=1,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=2
        )

        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc(x)

# Load transformer
input_dim = X.shape[1]
transformer_model = TransformerModel(input_dim)

transformer_model.load_state_dict(
    torch.load(os.path.join(MODEL_DIR, "solar_transformer_model.pth"), weights_only=True)
)

transformer_model.eval()

# ================= RF & XGB =================
rf_pred = rf_model.predict(X_test)
xgb_pred = xgb_model.predict(X_test)

# ================= LSTM =================
X_lstm = lstm_scaler.transform(X_test)

# reshape for LSTM
X_lstm = X_lstm.reshape((X_lstm.shape[0], 1, X_lstm.shape[1]))

lstm_pred_scaled = lstm_model.predict(X_lstm)

# ✅ FIX: Use same scaler properly (NOT manual scaling)
if hasattr(lstm_scaler, "inverse_transform"):
    try:
        lstm_pred = lstm_scaler.inverse_transform(lstm_pred_scaled)
    except:
        # fallback if scaler is only for X
        lstm_pred = lstm_pred_scaled
else:
    lstm_pred = lstm_pred_scaled


# ================= TRANSFORMER =================
X_trans = transformer_scalers["x_scaler"].transform(X_test)

X_tensor = torch.tensor(X_trans, dtype=torch.float32)
X_tensor = X_tensor.unsqueeze(1)

with torch.no_grad():
    transformer_pred_scaled = transformer_model(X_tensor).numpy()

# already correct
transformer_pred = transformer_scalers["y_scaler"].inverse_transform(transformer_pred_scaled)


# ================= FLATTEN =================
y_test = y_test.values.flatten()
rf_pred = rf_pred.flatten()
xgb_pred = xgb_pred.flatten()
lstm_pred = lstm_pred.flatten()
transformer_pred = transformer_pred.flatten()


# ================= ✅ CORRECT SCALE ALIGNMENT =================
global_min = min(
    y_test.min(),
    rf_pred.min(),
    xgb_pred.min(),
    lstm_pred.min(),
    transformer_pred.min()
)

global_max = max(
    y_test.max(),
    rf_pred.max(),
    xgb_pred.max(),
    lstm_pred.max(),
    transformer_pred.max()
)

def normalize_global(arr):
    return (arr - global_min) / (global_max - global_min + 1e-8)

y_plot = normalize_global(y_test)
rf_plot = normalize_global(rf_pred)
xgb_plot = normalize_global(xgb_pred)
lstm_plot = normalize_global(lstm_pred)
transformer_plot = normalize_global(transformer_pred)
# ================= DEBUG =================
print("\n===== FINAL DEBUG =====")
print("Actual:", y_test.min(), "to", y_test.max())
print("RF:", rf_pred.min(), "to", rf_pred.max())
print("XGB:", xgb_pred.min(), "to", xgb_pred.max())
print("LSTM:", lstm_pred.min(), "to", lstm_pred.max())
print("Transformer:", transformer_pred.min(), "to", transformer_pred.max())


# ================= PLOT (FIXED) =================
plt.figure(figsize=(14, 6))

plt.plot(y_plot[:500], label="Actual")
plt.plot(rf_plot[:500], label="Random Forest")
plt.plot(xgb_plot[:500], label="XGBoost")
plt.plot(lstm_plot[:500], label="LSTM")
plt.plot(transformer_plot[:500], label="Transformer")

plt.legend()
plt.title("Model Comparison (Fixed)")
plt.xlabel("Time")
plt.ylabel("Normalized AC Power")

plt.show()


print("\n========== NEXT HOUR PREDICTION ==========")

# KEEP dataframe format (important)
# pick a random non-zero power row
non_zero_indices = y[y > 200].index   # daytime data

sample_index = non_zero_indices[-10]  # or random.choice()

last_input_df = X.loc[[sample_index]]

print("\nUsing sample index:", sample_index)


# ================= RF & XGB =================
rf_next = rf_model.predict(last_input_df)[0]
xgb_next = xgb_model.predict(last_input_df)[0]

print("Random Forest:", rf_next)
print("XGBoost:", xgb_next)


# ================= LSTM =================
try:
    last_lstm = lstm_scaler.transform(last_input_df)
    last_lstm = last_lstm.reshape((1, 1, last_lstm.shape[1]))

    lstm_next = lstm_model.predict(last_lstm)

    # ❗ IMPORTANT: LSTM output is already near actual scale
    lstm_next = lstm_next[0][0]

    print("LSTM:", lstm_next)

except Exception as e:
    print("LSTM skipped:", e)


# ================= TRANSFORMER =================
try:
    last_trans = transformer_scalers["x_scaler"].transform(last_input_df)
    last_trans = torch.tensor(last_trans, dtype=torch.float32).unsqueeze(1)

    with torch.no_grad():
        trans_next = transformer_model(last_trans).numpy()

    trans_next = transformer_scalers["y_scaler"].inverse_transform(trans_next)[0][0]

    print("Transformer:", trans_next)

except Exception as e:
    print("Transformer skipped:", e)

