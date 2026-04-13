import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from data_preprocessing import load_and_preprocess_data

# ================= LOAD DATA =================
df, X, y, features = load_and_preprocess_data()

print("Training data shape:", df.shape)

# ================= SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# ================= MODEL =================
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

print("Training model...")
model.fit(X_train, y_train)

# ================= SAVE MODEL =================
model_dir = os.path.join(
    os.getcwd(),
    "renewable_energy_forecasting",
    "models"
)

os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "solar_model.pkl")

joblib.dump(model, model_path)

print("Model saved at:", model_path)