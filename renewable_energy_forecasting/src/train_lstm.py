import os
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

from data_preprocessing import load_and_preprocess_data


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

# ================= SCALE =================
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler (IMPORTANT)
model_dir = os.path.join(os.getcwd(), "renewable_energy_forecasting", "models")
os.makedirs(model_dir, exist_ok=True)

scaler_path = os.path.join(model_dir, "lstm_scaler.pkl")
joblib.dump(scaler, scaler_path)

# ================= RESHAPE =================
X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# ================= SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, shuffle=False
)

# ================= MODEL =================
model = Sequential()
model.add(LSTM(64, activation="relu", input_shape=(1, X_scaled.shape[2])))
model.add(Dense(32, activation="relu"))
model.add(Dense(1))

model.compile(optimizer="adam", loss="mse")

print("Training LSTM model...")

model.fit(
    X_train,
    y_train,
    epochs=15,
    batch_size=32,
    verbose=1
)

# ================= SAVE MODEL =================
model_path = os.path.join(model_dir, "solar_lstm_model.h5")
model.save(model_path)

print("LSTM model saved at:", model_path)