"""
Data preparation, feature engineering, and chronological splitting for Metro Interstate Traffic Volume dataset.
Strictly prevents data leakage by calculating imputation and scaling parameters solely on the training partition.
"""

from typing import Tuple, List, Dict, Any
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def prepare_traffic_data(
    file_path: str,
    test_size: float = 0.20,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[int], Dict[str, Any]]:
    """
    Loads, cleans, engineers features, and chronologically splits the traffic dataset.
    
    Returns:
        - X_train, y_train, X_test, y_test: Processed arrays.
        - feature_names: List of all column names in X.
        - continuous_indices: Indices of the continuous features (used for polynomial expansion).
        - metadata: Dictionary documenting split sizes, imputation values, and feature mappings.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Regression dataset not found: {file_path}")
        
    # keep_default_na=False so that holiday='None' is read as string, not NaN
    df = pd.read_csv(file_path, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    
    # 1. Parse timestamps and sort chronologically
    df["date_time"] = pd.to_datetime(df["date_time"])
    df = df.sort_values("date_time").reset_index(drop=True)
    
    total_samples = len(df)
    split_idx = int(total_samples * (1.0 - test_size))
    
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    # 2. Temperature 0.0 K anomaly inspection & imputation
    # Compute monthly median temperatures strictly from the training set
    train_df["month"] = train_df["date_time"].dt.month
    test_df["month"] = test_df["date_time"].dt.month
    
    # Valid positive temperatures in Kelvin
    valid_train_temps = train_df[train_df["temp"] > 0]
    monthly_temp_medians = valid_train_temps.groupby("month")["temp"].median().to_dict()
    overall_train_temp_median = float(valid_train_temps["temp"].median())
    
    # Impute 0 K entries
    train_zero_temps = int((train_df["temp"] == 0).sum())
    test_zero_temps = int((test_df["temp"] == 0).sum())
    
    for month_val, median_val in monthly_temp_medians.items():
        train_df.loc[(train_df["temp"] == 0) & (train_df["month"] == month_val), "temp"] = median_val
        test_df.loc[(test_df["temp"] == 0) & (test_df["month"] == month_val), "temp"] = median_val
        
    # Fallback in case of remaining zeros
    train_df.loc[train_df["temp"] == 0, "temp"] = overall_train_temp_median
    test_df.loc[test_df["temp"] == 0, "temp"] = overall_train_temp_median
    
    # 3. Rain extreme outlier handling: Cap extreme sensor artifact (9831.3 mm)
    # Threshold based on 99.99th percentile of training data
    rain_cap = float(train_df["rain_1h"].quantile(0.9999))
    train_df["rain_1h"] = np.minimum(train_df["rain_1h"], rain_cap)
    test_df["rain_1h"] = np.minimum(test_df["rain_1h"], rain_cap)
    
    # 4. Feature Engineering
    for split_df in [train_df, test_df]:
        # Temporal features
        hour = split_df["date_time"].dt.hour
        dayofweek = split_df["date_time"].dt.dayofweek
        month = split_df["date_time"].dt.month
        
        # Cyclical sine and cosine encodings
        split_df["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
        split_df["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
        split_df["dow_sin"] = np.sin(2.0 * np.pi * dayofweek / 7.0)
        split_df["dow_cos"] = np.cos(2.0 * np.pi * dayofweek / 7.0)
        split_df["month_sin"] = np.sin(2.0 * np.pi * month / 12.0)
        split_df["month_cos"] = np.cos(2.0 * np.pi * month / 12.0)
        
        # Indicator flags
        split_df["is_weekend"] = (dayofweek >= 5).astype(float)
        split_df["is_holiday"] = (split_df["holiday"] != "None").astype(float)
        
    # 5. Continuous Features Standardization (fit strictly on train)
    continuous_cols = ["temp", "rain_1h", "snow_1h", "clouds_all"]
    scaler = StandardScaler()
    scaler.fit(train_df[continuous_cols])
    
    train_cont_scaled = scaler.transform(train_df[continuous_cols])
    test_cont_scaled = scaler.transform(test_df[continuous_cols])
    
    # 6. Categorical One-Hot Encoding for weather_main (fit strictly on train)
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    ohe.fit(train_df[["weather_main"]])
    
    train_weather_ohe = ohe.transform(train_df[["weather_main"]])
    test_weather_ohe = ohe.transform(test_df[["weather_main"]])
    weather_feature_names = [f"weather_{c}" for c in ohe.categories_[0]]
    
    # 7. Assemble Feature Matrices
    engineered_cols = [
        "hour_sin", "hour_cos",
        "dow_sin", "dow_cos",
        "month_sin", "month_cos",
        "is_weekend", "is_holiday"
    ]
    train_eng = train_df[engineered_cols].to_numpy()
    test_eng = test_df[engineered_cols].to_numpy()
    
    # Continuous features are placed first so their column indices are [0, 1, 2, 3]
    X_train = np.hstack([train_cont_scaled, train_eng, train_weather_ohe])
    X_test = np.hstack([test_cont_scaled, test_eng, test_weather_ohe])
    
    all_feature_names = continuous_cols + engineered_cols + weather_feature_names
    continuous_indices = list(range(len(continuous_cols)))
    
    y_train = train_df["traffic_volume"].to_numpy(dtype=np.float64)
    y_test = test_df["traffic_volume"].to_numpy(dtype=np.float64)
    
    metadata = {
        "total_samples": total_samples,
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "train_date_range": [str(train_df["date_time"].min()), str(train_df["date_time"].max())],
        "test_date_range": [str(test_df["date_time"].min()), str(test_df["date_time"].max())],
        "zero_temp_imputed_train": train_zero_temps,
        "zero_temp_imputed_test": test_zero_temps,
        "rain_cap_mm": rain_cap,
        "num_features": len(all_feature_names),
        "continuous_features": continuous_cols
    }
    
    return X_train, y_train, X_test, y_test, all_feature_names, continuous_indices, metadata
