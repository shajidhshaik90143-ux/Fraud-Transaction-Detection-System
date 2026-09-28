from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

DATA_FILE = DATA_DIR / "transactions.csv"
MODEL_FILE = MODEL_DIR / "fraud_pipeline.joblib"
METADATA_FILE = MODEL_DIR / "model_metadata.json"

RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "amount", "transaction_hour", "customer_age", "account_age_days",
    "transactions_last_24h", "avg_amount_30d", "distance_from_home_km",
    "device_trust_score", "failed_login_attempts", "is_international",
    "card_present", "is_new_device",
]
CATEGORICAL_FEATURES = ["merchant_category"]
TARGET = "is_fraud"
ID_COLUMN = "transaction_id"
REQUIRED_PREDICTION_COLUMNS = [ID_COLUMN] + NUMERIC_FEATURES + CATEGORICAL_FEATURES
