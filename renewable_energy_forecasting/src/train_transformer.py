import os
import torch
import torch.nn as nn
import numpy as np

from sklearn.preprocessing import MinMaxScaler

from data_preprocessing import load_and_preprocess_data   # ✅ FIXED


# ================= LOAD DATA =================
df,X,y,features = load_and_preprocess_data()

# ================= FEATURES =================
features = [
    "DC_POWER",
    "AMBIENT_TEMPERATURE",
    "MODULE_TEMPERATURE",
    "IRRADIATION",
    "GHI",
    "DNI",
    "DHI",
    "Temperature",
    "Wind Speed",
    "lag_1",
    "lag_2",
    "lag_3"
]

target = "AC_POWER"


y = df[target]

# Reduce size for speed
X = X.iloc[:10000]
y = y.iloc[:10000]

# ================= SCALE =================
x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

X_scaled = x_scaler.fit_transform(X)
y_scaled = y_scaler.fit_transform(y.values.reshape(-1, 1))

# ================= TENSOR =================
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

# Add sequence dimension
X_tensor = X_tensor.unsqueeze(1)

# ================= SPLIT =================
train_size = int(0.8 * len(X_tensor))

X_train = X_tensor[:train_size]
y_train = y_tensor[:train_size]

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


model = TransformerModel(X_train.shape[2])

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Training Transformer model...")

# ================= TRAIN =================
epochs = 60

for epoch in range(epochs):
    model.train()

    optimizer.zero_grad()

    output = model(X_train)

    loss = criterion(output, y_train)

    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# ================= SAVE =================
base_dir = os.path.dirname(os.path.dirname(__file__))

model_path = os.path.join(base_dir, "models", "solar_transformer_model.pth")
scaler_path = os.path.join(base_dir, "models", "transformer_scalers.pth")

torch.save(model.state_dict(), model_path)

torch.save({
    "x_scaler": x_scaler,
    "y_scaler": y_scaler
}, scaler_path)

print("Transformer model training complete")
