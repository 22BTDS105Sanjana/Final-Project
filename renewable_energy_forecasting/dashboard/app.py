import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import os
import sys
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# --- PATH SETUP ---
curr_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(curr_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_preprocessing import load_and_preprocess_data

# --- APP CONFIG ---
st.set_page_config(page_title="Solar AI Forecast", layout="wide")
MODEL_DIR = os.path.join(project_root, "models")

# --- TRANSFORMER CLASS ---
class TransformerModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=1, dim_feedforward=256, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        self.fc = nn.Linear(input_dim, 1)
    def forward(self, x):
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc(x)

# --- DATA LOADING ---
@st.cache_data
def get_data():
    df, X, y, features = load_and_preprocess_data()
    # Using daylight data for the visual test set
    day_df = df[df['IRRADIATION'] > 0.1].tail(500) 
    X_test = X.loc[day_df.index]
    y_test = y.loc[day_df.index]
    return day_df, X_test, y_test, features

df_display, X_test, y_test, features = get_data()

# --- PREDICTION ENGINE ---
def run_prediction(name, data):
    if name == "Random Forest":
        m = joblib.load(os.path.join(MODEL_DIR, "solar_model.pkl"))
        return m.predict(data)
    
    elif name == "XGBoost":
        m = joblib.load(os.path.join(MODEL_DIR, "solar_xgb_model.pkl"))
        return m.predict(data)
    
    elif name == "LSTM":
        m = load_model(os.path.join(MODEL_DIR, "solar_lstm_model.h5"), compile=False)
        sc = joblib.load(os.path.join(MODEL_DIR, "lstm_scaler.pkl"))
        scaled_x = sc.transform(data)
        reshaped_x = scaled_x.reshape((scaled_x.shape[0], 1, scaled_x.shape[1]))
        return m.predict(reshaped_x, verbose=0).flatten()
        
    elif name == "Transformer":
        scs = torch.load(os.path.join(MODEL_DIR, "transformer_scalers.pth"), weights_only=False)
        m = TransformerModel(len(features))
        m.load_state_dict(torch.load(os.path.join(MODEL_DIR, "solar_transformer_model.pth"), weights_only=True))
        m.eval()
        scaled_x = scs["x_scaler"].transform(data)
        tensor_x = torch.tensor(scaled_x, dtype=torch.float32).unsqueeze(1)
        with torch.no_grad():
            raw_p = m(tensor_x).numpy()
        return scs["y_scaler"].inverse_transform(raw_p).flatten()

# --- SIDEBAR ---
st.sidebar.title("🚀 Navigation")
page = st.sidebar.radio("View Selection", ["Individual Analysis", "Visualize All Models", "Next Hour MAPIE"])

# --- PAGE 1: INDIVIDUAL ANALYSIS ---
if page == "Individual Analysis":
    st.title("🔎 Model Evaluation Metrics")
    sel = st.selectbox("Select Model", ["Random Forest", "XGBoost", "LSTM", "Transformer"])
    p = run_prediction(sel, X_test)
    y_true = y_test.values
    
    rmse = np.sqrt(mean_squared_error(y_true, p))
    mae = mean_absolute_error(y_true, p)
    r2 = r2_score(y_true, p)
    
    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE", f"{rmse:.2f}")
    c2.metric("MAE", f"{mae:.2f}")
    c3.metric("R² Score", f"{r2:.4f}")
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(y_true[:100], label="Actual (kW)", color="black", alpha=0.8)
    ax.plot(p[:100], label=f"{sel} Prediction", color="orange", linestyle="--")
    ax.set_ylabel("Power Output (kW)")
    ax.legend()
    st.pyplot(fig)

# --- PAGE 2: VISUALIZE ALL MODELS ---
elif page == "Visualize All Models":
    st.title("📊 Global Comparison")
    with st.spinner("Calculating..."):
        rf_p = run_prediction("Random Forest", X_test)
        xgb_p = run_prediction("XGBoost", X_test)
        lstm_p = run_prediction("LSTM", X_test)
        trans_p = run_prediction("Transformer", X_test)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(y_test.values[:100], label="Actual", color="black", linewidth=2)
    ax.plot(rf_p[:100], label="RF", alpha=0.7)
    ax.plot(xgb_p[:100], label="XGB", alpha=0.7)
    ax.plot(lstm_p[:100], label="LSTM", alpha=0.7)
    ax.plot(trans_p[:100], label="Transformer", alpha=0.7)
    ax.legend()
    st.pyplot(fig)

# --- PAGE 3: CORRECTED MAPIE SECTION ---
elif page == "Next Hour MAPIE":
    st.title("🔮 Next Hour MAPIE Forecast")
    
    # 1. Prediction using the Ensemble (or your chosen Base model)
    # Most users prefer the Ensemble for MAPIE to capture all model variance
    with st.spinner("Generating Ensemble Forecast..."):
        p_xgb = run_prediction("XGBoost", X_test)
        p_lstm = run_prediction("LSTM", X_test)
        p_trans = run_prediction("Transformer", X_test)
        ensemble_preds = (p_xgb + p_lstm + p_trans) / 3

    # 2. Conformal Interval calculation (Correct MAPIE Logic)
    residuals = np.abs(y_test.values - ensemble_preds)
    q = np.quantile(residuals, 0.95) # 95% Confidence Interval

    # 3. Next Hour Metrics
    prediction = ensemble_preds[-1]
    st.metric("Next Hour Forecast", f"{prediction:.2f} kW")
    st.write(f"95% Confidence Interval: **{max(0, prediction-q):.2f} to {prediction+q:.2f} kW**")
    
    # 4. Corrected Visualization
    fig, ax = plt.subplots(figsize=(12, 5))
    # Show last 100 actual points
    ax.plot(y_test.values[-100:], label="Actual", color="black", alpha=0.5)
    # Show ensemble prediction
    ax.plot(ensemble_preds[-100:], label="Ensemble Prediction", color="red")
    # Fill interval
    ax.fill_between(range(100), 
                     ensemble_preds[-100:] - q, 
                     ensemble_preds[-100:] + q, 
                     color='red', alpha=0.2, label="Prediction Interval")
    ax.set_title("MAPIE: Uncertainty Quantification")
    ax.legend()
    st.pyplot(fig)