import json
import pandas as pd
import streamlit as st
from src.config import *
from src.predict import predict_dataframe, explain_row

st.set_page_config(page_title="Fraud Transaction Detection", page_icon="🛡️", layout="wide")
st.title("🛡️ Fraud Transaction Detection System")
st.caption("End-to-end machine-learning dashboard for potentially fraudulent transactions.")

if not MODEL_FILE.exists():
    st.error("Model is missing. Run `python src/train.py` first.")
    st.stop()

metrics_file = REPORT_DIR/"metrics.json"
metrics = json.loads(metrics_file.read_text()) if metrics_file.exists() else {}
page = st.sidebar.radio("Module", ["Dashboard","Single Transaction","Batch CSV","Model Reports","About"])

if page == "Dashboard":
    st.subheader("System Overview")
    if DATA_FILE.exists():
        df = pd.read_csv(DATA_FILE)
        a,b,c,d = st.columns(4)
        a.metric("Transactions", f"{len(df):,}")
        b.metric("Fraud Rate", f"{df.is_fraud.mean():.2%}")
        c.metric("Average Amount", f"${df.amount.mean():,.2f}")
        d.metric("ROC-AUC", f"{metrics.get('roc_auc',0):.3f}")
        x,y = st.columns(2)
        with x:
            st.write("### Fraud Distribution")
            st.bar_chart(df.is_fraud.value_counts().rename({0:"Legitimate",1:"Fraud"}))
        with y:
            st.write("### Merchant Categories")
            st.bar_chart(df.merchant_category.value_counts())
        st.write("### Transaction Amounts")
        st.line_chart(df.amount.clip(upper=df.amount.quantile(.99)).head(500).reset_index(drop=True))

elif page == "Single Transaction":
    st.subheader("Analyze a Transaction")
    with st.form("tx"):
        a,b,c = st.columns(3)
        with a:
            tid = st.text_input("Transaction ID","TXN-DEMO-001")
            amount = st.number_input("Amount",1.0,100000.0,250.0)
            hour = st.slider("Transaction Hour",0,23,14)
            merchant = st.selectbox("Merchant",["grocery","electronics","restaurant","travel","fuel","online_shopping","pharmacy","gaming","jewelry","cash_withdrawal"])
            age = st.number_input("Customer Age",18,100,30)
        with b:
            account_age = st.number_input("Account Age (days)",1,10000,800)
            tx24 = st.number_input("Transactions Last 24h",0,100,2)
            avg30 = st.number_input("Average Amount (30d)",1.0,100000.0,180.0)
            distance = st.number_input("Distance From Home (km)",0.0,10000.0,5.0)
            trust = st.slider("Device Trust Score",0.0,100.0,85.0)
        with c:
            failed = st.number_input("Failed Login Attempts",0,30,0)
            intl = st.selectbox("International",[0,1],format_func=lambda x:"Yes" if x else "No")
            card = st.selectbox("Card Present",[1,0],format_func=lambda x:"Yes" if x else "No")
            new = st.selectbox("New Device",[0,1],format_func=lambda x:"Yes" if x else "No")
        submitted = st.form_submit_button("🔍 Analyze", use_container_width=True)
    if submitted:
        row = pd.DataFrame([{
            "transaction_id":tid,"amount":amount,"transaction_hour":hour,
            "merchant_category":merchant,"customer_age":age,"account_age_days":account_age,
            "transactions_last_24h":tx24,"avg_amount_30d":avg30,
            "distance_from_home_km":distance,"device_trust_score":trust,
            "failed_login_attempts":failed,"is_international":intl,
            "card_present":card,"is_new_device":new
        }])
        result = predict_dataframe(row).iloc[0]
        risk = float(result.risk_score)
        if risk >= 75: st.error(f"🚨 {result.prediction_label} — Critical Risk")
        elif risk >= 50: st.warning(f"⚠️ {result.prediction_label} — High Risk")
        elif risk >= 20: st.info(f"ℹ️ {result.prediction_label} — Medium Risk")
        else: st.success(f"✅ {result.prediction_label} — Low Risk")
        x,y,z = st.columns(3)
        x.metric("Fraud Probability",f"{result.fraud_probability:.2%}")
        y.metric("Risk Score",f"{result.risk_score:.2f}/100")
        z.metric("Decision",result.prediction_label)
        st.write("### Risk Indicators")
        for r in explain_row(result): st.write("• " + r)

elif page == "Batch CSV":
    st.subheader("Batch Prediction")
    st.write("Required columns:")
    st.code(", ".join(REQUIRED_PREDICTION_COLUMNS))
    uploaded = st.file_uploader("Upload CSV", type="csv")
    if uploaded:
        try:
            inp = pd.read_csv(uploaded)
            result = predict_dataframe(inp)
            st.dataframe(result, use_container_width=True)
            n = int(result.prediction.sum())
            a,b,c = st.columns(3)
            a.metric("Rows",len(result)); b.metric("Flagged Fraud",n); c.metric("Flag Rate",f"{n/max(1,len(result)):.2%}")
            st.download_button("⬇️ Download Predictions", result.to_csv(index=False).encode(), "fraud_predictions.csv", "text/csv", use_container_width=True)
        except Exception as e:
            st.error(str(e))

elif page == "Model Reports":
    st.subheader("Evaluation")
    if metrics:
        cols = st.columns(5)
        for col,key in zip(cols,["accuracy","precision","recall","f1","pr_auc"]):
            col.metric(key.upper(),f"{metrics[key]:.3f}")
    for fn,title in [("confusion_matrix.png","Confusion Matrix"),("roc_curve.png","ROC Curve"),
                     ("precision_recall_curve.png","Precision-Recall Curve"),("feature_importance.png","Feature Importance")]:
        p = REPORT_DIR/fn
        if p.exists():
            st.write("### "+title); st.image(str(p), use_container_width=True)

elif page == "About":
    st.subheader("About")
    st.markdown("""This project demonstrates a complete fraud-detection workflow:
data generation → preprocessing → Random Forest → probability → risk score → dashboard.

The data is synthetic. A real financial deployment would require privacy/security controls,
model monitoring, calibration, drift detection, audit logs and human review.""")

st.divider()
st.caption("Educational / demonstration project — model output is a risk signal, not proof of fraud.")
