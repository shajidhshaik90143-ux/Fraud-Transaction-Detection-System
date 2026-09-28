import json
import warnings
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .config import *

warnings.filterwarnings("ignore")

def build_pipeline():
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    pre = ColumnTransformer([("num", numeric, NUMERIC_FEATURES),
                             ("cat", categorical, CATEGORICAL_FEATURES)])
    clf = RandomForestClassifier(
        n_estimators=350, max_depth=14, min_samples_leaf=2,
        class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1
    )
    return Pipeline([("preprocessor", pre), ("classifier", clf)])

def main():
    if not DATA_FILE.exists():
        from generate_data import generate_transactions
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        generate_transactions().to_csv(DATA_FILE, index=False)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_FILE)
    X, y = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES], df[TARGET].astype(int)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=.30, stratify=y, random_state=RANDOM_STATE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=.50, stratify=y_temp, random_state=RANDOM_STATE)

    model = build_pipeline()
    model.fit(X_train, y_train)

    val_prob = model.predict_proba(X_val)[:, 1]
    thresholds = np.linspace(.05, .95, 181)
    scores = [f1_score(y_val, val_prob >= t, zero_division=0) for t in thresholds]
    threshold = float(thresholds[int(np.argmax(scores))])

    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= threshold).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, prob)),
        "pr_auc": float(average_precision_score(y_test, prob)),
        "decision_threshold": threshold,
        "train_rows": len(X_train), "validation_rows": len(X_val),
        "test_rows": len(X_test), "fraud_rate": float(y.mean())
    }
    joblib.dump(model, MODEL_FILE)
    (REPORT_DIR/"metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (MODEL_DIR/"model_metadata.json").write_text(json.dumps({
        "model": "RandomForestClassifier", "threshold": threshold,
        "features": NUMERIC_FEATURES + CATEGORICAL_FEATURES, "target": TARGET
    }, indent=2), encoding="utf-8")

    cm = confusion_matrix(y_test, pred)
    plt.figure(figsize=(6,5)); sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Legitimate","Fraud"], yticklabels=["Legitimate","Fraud"])
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion Matrix")
    plt.tight_layout(); plt.savefig(REPORT_DIR/"confusion_matrix.png", dpi=160); plt.close()

    fpr, tpr, _ = roc_curve(y_test, prob)
    plt.figure(figsize=(7,5)); plt.plot(fpr,tpr,label=f"ROC-AUC={metrics['roc_auc']:.3f}")
    plt.plot([0,1],[0,1],"--"); plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve"); plt.legend(); plt.tight_layout()
    plt.savefig(REPORT_DIR/"roc_curve.png", dpi=160); plt.close()

    precision, recall, _ = precision_recall_curve(y_test, prob)
    plt.figure(figsize=(7,5)); plt.plot(recall,precision,label=f"PR-AUC={metrics['pr_auc']:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall Curve")
    plt.legend(); plt.tight_layout(); plt.savefig(REPORT_DIR/"precision_recall_curve.png", dpi=160); plt.close()

    names = model.named_steps["preprocessor"].get_feature_names_out()
    imp = model.named_steps["classifier"].feature_importances_
    order = np.argsort(imp)[-20:]
    plt.figure(figsize=(9,7)); plt.barh(range(len(order)), imp[order])
    plt.yticks(range(len(order)), names[order]); plt.xlabel("Importance")
    plt.title("Top Model Features"); plt.tight_layout()
    plt.savefig(REPORT_DIR/"feature_importance.png", dpi=160); plt.close()

    (REPORT_DIR/"classification_report.txt").write_text(
        classification_report(y_test, pred, target_names=["Legitimate","Fraud"]), encoding="utf-8")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
