import pandas as pd
import os

def load_and_preprocess_data():
    # ================= FIXED PATH LOGIC =================
    # This gets the directory where data_preprocessing.py is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # This goes UP one level to the root, then into the 'data' folder
    base_path = os.path.join(os.path.dirname(current_dir), "data")

    gen_path = os.path.join(base_path, "Plant_1_Generation_Data.csv")
    weather_path = os.path.join(base_path, "Plant_1_Weather_Sensor_Data.csv")

    # Debug prints to verify the path is correct in your terminal
    print(f"Looking for Gen Data at: {gen_path}")
    
    if not os.path.exists(gen_path):
        raise FileNotFoundError(f"STILL NOT FOUND: Check if {gen_path} exists.")

    # ================= LOAD =================
    gen_df = pd.read_csv(gen_path)
    weather_df = pd.read_csv(weather_path)

    # ================= FIX DATE =================
    gen_df["DATE_TIME"] = pd.to_datetime(gen_df["DATE_TIME"], dayfirst=True)
    weather_df["DATE_TIME"] = pd.to_datetime(weather_df["DATE_TIME"], dayfirst=True)

    # ================= MERGE =================
    df = pd.merge(gen_df, weather_df, on="DATE_TIME")

    print("After merge shape:", df.shape)

    # ================= CREATE EXTRA FEATURES =================
    # (since NSRDB removed)
    df["GHI"] = df["IRRADIATION"]
    df["DNI"] = df["IRRADIATION"] * 0.7
    df["DHI"] = df["IRRADIATION"] * 0.3

    df["Temperature"] = df["AMBIENT_TEMPERATURE"]
    df["Wind Speed"] = 0  # not available

    # ================= LAG FEATURES =================
    df["lag_1"] = df["AC_POWER"].shift(1)
    df["lag_2"] = df["AC_POWER"].shift(2)
    df["lag_3"] = df["AC_POWER"].shift(3)

    # ================= DROP NA =================
    df = df.dropna(subset=["lag_1", "lag_2", "lag_3"])

    print("Final dataset shape:", df.shape)
    # ================= SORT TIME =================
    df = df.sort_values("DATE_TIME")
    df = df.reset_index(drop=True)
    
   # ================= TARGET =================
    y = df["AC_POWER"]

   # ================= FEATURES =================
   # 🚨 IMPORTANT: Keep only useful numeric features (same as training)
    feature_cols = [
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

    X = df[feature_cols]

    print("Final dataset shape:", df.shape)

    return df, X, y, feature_cols
