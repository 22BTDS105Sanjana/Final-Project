import os
import joblib

from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from data_preprocessing import load_and_preprocess_data

# ================= LOAD DATA =================
df, X, y, features = load_and_preprocess_data()

print("Training data shape:", df.shape)

# ================= SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# ================= MODEL =================
model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

print("Training XGBoost model...")
model.fit(X_train, y_train)

# ================= SAVE MODEL =================
model_dir = os.path.join(
    os.getcwd(),
    "renewable_energy_forecasting",
    "models"
)

os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "solar_xgb_model.pkl")

joblib.dump(model, model_path)

print("XGBoost model saved at:", model_path)