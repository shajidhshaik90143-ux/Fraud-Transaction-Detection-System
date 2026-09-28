import pandas as pd
from .config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, REQUIRED_PREDICTION_COLUMNS

def validate_prediction_frame(df):
    missing = [c for c in REQUIRED_PREDICTION_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

def prepare_features(df):
    validate_prediction_frame(df)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    for col in NUMERIC_FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    if X[NUMERIC_FEATURES].isna().any().any():
        bad = X[NUMERIC_FEATURES].columns[X[NUMERIC_FEATURES].isna().any()].tolist()
        raise ValueError("Invalid numeric values in: " + ", ".join(bad))
    X["merchant_category"] = X["merchant_category"].astype(str).str.strip().str.lower()
    return X
