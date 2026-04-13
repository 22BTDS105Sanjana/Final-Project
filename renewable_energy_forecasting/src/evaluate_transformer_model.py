import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_preprocessing import load_and_preprocess_data


# ================= LOAD DATA =================
df, X, y, features = load_and_preprocess_data()

# (optional limit for faster testing)
X = X.iloc[:10000]
y = y.iloc[:10000]

# ================= LOAD SCALERS =================
base_dir = os.path.dirname(os.path.dirname(__file__))

scaler_path = os.path.join(base_dir, "models", "transformer_scalers.pth")
scalers = torch.load(scaler_path, weights_only=False)

x_scaler = scalers["x_scaler"]
y_scaler = scalers["y_scaler"]

# ================= SCALE =================
X_scaled = x_scaler.transform(X)

# ================= TENSOR =================
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
X_tensor = X_tensor.unsqueeze(1)

# ================= SPLIT =================
train_size = int(0.8 * len(X_tensor))

X_test = X_tensor[train_size:]
y_test = y.values[train_size:]

# ================= MODEL =================
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


model = TransformerModel(X_tensor.shape[2])

model_path = os.path.join(base_dir, "models", "solar_transformer_model.pth")
model.load_state_dict(torch.load(model_path))

model.eval()

print("Transformer model loaded")

# ================= PREDICT =================
with torch.no_grad():
    preds_scaled = model(X_test).numpy()

# ================= INVERSE SCALE =================
predictions = y_scaler.inverse_transform(preds_scaled).flatten()
y_test = y_test.flatten()

# ================= METRICS =================
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\nTransformer Performance:")
print("MAE :", mae)
print("RMSE:", rmse)
print("R2  :", r2)

# ================= SMOOTH GRAPH =================
actual_series = pd.Series(y_test)
pred_series = pd.Series(predictions)

smooth_actual = actual_series.rolling(window=5).mean()
smooth_pred = pred_series.rolling(window=5).mean()

# ================= PLOT =================
plt.figure(figsize=(12,6))

plt.plot(smooth_actual[:200], label="Actual", linewidth=2)
plt.plot(smooth_pred[:200], label="Predicted", linestyle="--")

plt.title("Solar Power Prediction (Transformer - Smoothed)")
plt.xlabel("Time Steps")
plt.ylabel("AC Power")

plt.legend()
plt.grid()

plt.show()