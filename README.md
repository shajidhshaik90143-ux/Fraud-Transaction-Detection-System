# Fraud Transaction Detection System

A complete educational fraud-detection project using Python, scikit-learn and Streamlit.

## Features
- Synthetic 12,000-transaction dataset included
- Random Forest fraud classifier with class-imbalance handling
- Automatic validation threshold selection
- Precision, recall, F1, ROC-AUC and PR-AUC
- Confusion matrix, ROC curve, PR curve and feature importance
- Single-transaction risk scoring
- Batch CSV prediction and downloadable results
- Risk levels: Low / Medium / High / Critical
- Rule-based risk indicators for user-facing explanations
- Saved model pipeline and metadata
- Automated tests

## Quick start
Recommended Python: 3.11 or 3.12.

```bash
python -m venv .venv
```

Windows PowerShell:
```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/train.py
python -m streamlit run app.py
```

If the dataset is missing:
```bash
python src/generate_data.py
python src/train.py
```

Windows users can also run `run_project.bat`.

## CSV prediction columns
`transaction_id, amount, transaction_hour, merchant_category, customer_age, account_age_days, transactions_last_24h, avg_amount_30d, distance_from_home_km, device_trust_score, failed_login_attempts, is_international, card_present, is_new_device`

Training data additionally contains `is_fraud`.

## Project structure
```text
app.py
requirements.txt
data/
models/
reports/
src/
  config.py
  features.py
  generate_data.py
  predict.py
  train.py
tests/
```

## Disclaimer
The included data is synthetic and the project is for education/demo use. Real financial systems require domain validation, privacy controls, authentication, audit logging, model monitoring, drift detection, calibrated thresholds and human review.
