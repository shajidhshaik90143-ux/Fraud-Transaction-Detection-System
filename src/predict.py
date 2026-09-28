import json
import joblib
import pandas as pd
from .config import MODEL_FILE
from .features import prepare_features

def load_model():
    if not MODEL_FILE.exists():
        raise FileNotFoundError("Model not found. Run python src/train.py")
    model = joblib.load(MODEL_FILE)
    meta_file = MODEL_FILE.parent / "model_metadata.json"
    meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
    return model, meta

def predict_dataframe(df):
    model, meta = load_model()
    X = prepare_features(df)
    probability = model.predict_proba(X)[:,1]
    threshold = float(meta.get("threshold", .5))
    out = df.copy()
    out["fraud_probability"] = probability
    out["risk_score"] = (probability*100).round(2)
    out["risk_level"] = pd.cut(probability, [-.01,.20,.50,.75,1.01],
                               labels=["Low","Medium","High","Critical"]).astype(str)
    out["prediction"] = (probability >= threshold).astype(int)
    out["prediction_label"] = out.prediction.map({0:"Legitimate",1:"Fraud"})
    return out

def explain_row(row):
    reasons = []
    if float(row.amount) > max(500, float(row.avg_amount_30d)*3):
        reasons.append("Amount is unusually high versus recent average.")
    if int(row.is_new_device): reasons.append("New device detected.")
    if float(row.device_trust_score) < 45: reasons.append("Low device trust score.")
    if int(row.is_international): reasons.append("International transaction.")
    if int(row.failed_login_attempts) >= 2: reasons.append("Multiple failed login attempts.")
    if int(row.transactions_last_24h) >= 7: reasons.append("High recent transaction activity.")
    if float(row.distance_from_home_km) > 80: reasons.append("Transaction is far from home location.")
    if int(row.card_present) == 0: reasons.append("Card was not physically present.")
    return reasons or ["No major rule-based risk indicators were triggered."]
