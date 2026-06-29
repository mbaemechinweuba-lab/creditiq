import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import datetime
import json
 
# ── Firebase ──────────────────────────────────────────────────────────────────
import firebase_admin
from firebase_admin import credentials, firestore
 
@st.cache_resource
def init_firebase():
    try:
        if not firebase_admin._apps:
            fb = st.secrets["firebase"]
            # Handle private key — replace literal \n with real newlines
            private_key = fb["private_key"]
            if "\\n" in private_key:
                private_key = private_key.replace("\\n", "\n")
 
            cred_dict = {
                "type":                        str(fb["type"]),
                "project_id":                  str(fb["project_id"]),
                "private_key_id":              str(fb["private_key_id"]),
                "private_key":                 private_key,
                "client_email":                str(fb["client_email"]),
                "client_id":                   str(fb["client_id"]),
                "auth_uri":                    str(fb["auth_uri"]),
                "token_uri":                   str(fb["token_uri"]),
                "auth_provider_x509_cert_url": str(fb["auth_provider_x509_cert_url"]),
                "client_x509_cert_url":        str(fb["client_x509_cert_url"]),
                "universe_domain":             "googleapis.com",
            }
            cred = credentials.Certificate(cred_dict)
            firebase_admin.initialize_app(cred)
        return firestore.client(), None
    except Exception as e:
        return None, str(e)
 
def log_assessment(db, borrower_id, risk_label, default_prob, input_data):
    if db is None:
        return
    try:
        db.collection("assessments").add({
            "borrower_id":  borrower_id,
            "risk_level":   risk_label,
            "default_prob": round(float(default_prob), 2),
            "timestamp":    datetime.datetime.utcnow().isoformat(),
            "inputs": {
                "existing_debt":   int(input_data["existing_debt_amount"].iloc[0]),
                "savings_worth":   int(input_data["savings_worth"].iloc[0]),
                "loan_requested":  int(input_data["loan_amount_requested"].iloc[0]),
                "online_txns_30d": int(input_data["online_txn_count_last_30d"].iloc[0]),
                "finance_app_pct": float(input_data["finance_app_time_pct"].iloc[0]),
            }
        })
    except Exception as e:
        st.session_state["firebase_error"] = str(e)
 
# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CreditIQ — Digital Behavioural Credit Scoring",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)
 
# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0b0f14; color: #e2eaf4; }
    section[data-testid="stSidebar"] { background-color: #121920; border-right: 1px solid #1e2d3d; }
    section[data-testid="stSidebar"] * { color: #e2eaf4 !important; }
    h1, h2, h3 { color: #00c896 !important; font-family: 'IBM Plex Mono', monospace; }
    div[data-testid="metric-container"] { background-color: #121920; border: 1px solid #1e2d3d; border-radius: 8px; padding: 16px; }
    div[data-testid="metric-container"] label { color: #5a7a99 !important; font-size: 11px; }
    div[data-testid="metric-container"] div { color: #e2eaf4 !important; }
    .verdict-low { background: linear-gradient(135deg, #121920, rgba(0,200,150,0.08)); border: 2px solid #00c896; border-radius: 10px; padding: 24px 28px; text-align: center; }
    .verdict-high { background: linear-gradient(135deg, #121920, rgba(255,77,109,0.08)); border: 2px solid #ff4d6d; border-radius: 10px; padding: 24px 28px; text-align: center; }
    .verdict-medium { background: linear-gradient(135deg, #121920, rgba(245,158,11,0.08)); border: 2px solid #f59e0b; border-radius: 10px; padding: 24px 28px; text-align: center; }
    .verdict-title-low  { font-size: 32px; font-weight: 700; color: #00c896; font-family: monospace; }
    .verdict-title-high { font-size: 32px; font-weight: 700; color: #ff4d6d; font-family: monospace; }
    .verdict-title-med  { font-size: 32px; font-weight: 700; color: #f59e0b; font-family: monospace; }
    .verdict-sub { font-size: 13px; color: #8ba3bd; margin-top: 6px; }
    .section-label { font-family: monospace; font-size: 10px; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: #5a7a99; margin-bottom: 8px; }
    .dataframe { background-color: #121920 !important; color: #e2eaf4 !important; }
    .stButton > button { background-color: #00c896 !important; color: #000 !important; font-weight: 700 !important; font-family: monospace !important; letter-spacing: 0.06em !important; border: none !important; border-radius: 4px !important; width: 100% !important; padding: 12px !important; font-size: 13px !important; }
    .stButton > button:hover { opacity: 0.85 !important; }
    hr { border-color: #1e2d3d !important; }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)
 
# ── Initialise Firebase ───────────────────────────────────────────────────────
db, firebase_err = init_firebase()
 
# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model  = joblib.load("creditiq_model.pkl")
        scaler = joblib.load("creditiq_scaler.pkl")
        return model, scaler
    except Exception:
        return None, None
 
model, scaler = load_model()
 
# ── Header ────────────────────────────────────────────────────────────────────
col_logo, col_status = st.columns([3, 1])
with col_logo:
    st.markdown("# 💳 CreditIQ")
    st.markdown("<p style='color:#5a7a99;font-family:monospace;font-size:13px;margin-top:-12px;'>Digital Behavioural Credit Scoring System &nbsp;|&nbsp; Mbaeme Chinweuba Ronald &nbsp;|&nbsp; 22-10448</p>", unsafe_allow_html=True)
with col_status:
    if model is not None:
        st.success("🟢 Model Loaded")
        if db is not None:
            st.success("🔥 Firebase Connected")
        else:
            st.warning(f"⚠️ Firebase: {firebase_err[:40] if firebase_err else 'Not connected'}...")
    else:
        st.error("🔴 Model Not Found")
 
st.markdown("---")
 
if model is None:
    st.warning("⚠️ Model files not found. Upload `creditiq_model.pkl` and `creditiq_scaler.pkl`.")
    st.stop()
 
# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📋 Borrower Profile")
    st.markdown("<div class='section-label'>Identity</div>", unsafe_allow_html=True)
    borrower_id = st.text_input("Borrower ID", value="NG-2024-00001")
 
    st.markdown("---")
    st.markdown("<div class='section-label'>Mobile & App Usage</div>", unsafe_allow_html=True)
    unique_apps_per_day       = st.number_input("Unique Apps Per Day",         min_value=0,   max_value=50,    value=8)
    avg_daily_screen_time_hrs = st.number_input("Avg Daily Screen Time (hrs)", min_value=0.0, max_value=24.0,  value=4.5, step=0.5)
    financial_apps_installed  = st.number_input("Financial Apps Installed",    min_value=0,   max_value=20,    value=3)
    online_txn_count_last_30d = st.number_input("Online Transactions (30d)",   min_value=0,   max_value=200,   value=18)
    avg_txn_amount            = st.number_input("Avg Transaction Amount (₦)",  min_value=0,   max_value=500000, value=12000, step=500)
 
    st.markdown("---")
    st.markdown("<div class='section-label'>Communication Patterns</div>", unsafe_allow_html=True)
    bank_sms_count           = st.number_input("Bank SMS Count (30d)",      min_value=0,   max_value=200,  value=12)
    calls_per_day            = st.number_input("Calls Per Day",             min_value=0,   max_value=100,  value=6)
    distinct_contacts_weekly = st.number_input("Distinct Contacts Weekly",  min_value=0,   max_value=500,  value=24)
    avg_call_duration_mins   = st.number_input("Avg Call Duration (mins)",  min_value=0.0, max_value=60.0, value=3.5, step=0.5)
 
    st.markdown("---")
    st.markdown("<div class='section-label'>Lifestyle & Behaviour</div>", unsafe_allow_html=True)
    avg_distance_travelled_km = st.number_input("Avg Distance Travelled (km/day)", min_value=0.0, max_value=200.0, value=12.0, step=0.5)
    places_visited_weekly     = st.number_input("Places Visited Weekly",            min_value=0,   max_value=50,   value=7)
    social_media_pct          = st.number_input("Social Media Usage (%)",           min_value=0.0, max_value=100.0, value=25.0, step=1.0)
    finance_app_time_pct      = st.number_input("Finance App Time (%)",             min_value=0.0, max_value=100.0, value=15.0, step=1.0)
    regular_sleep_pattern     = st.selectbox("Regular Sleep Pattern",    [1, 0], format_func=lambda x: "Yes" if x==1 else "No")
    battery_charging_regular  = st.selectbox("Regular Battery Charging", [1, 0], format_func=lambda x: "Yes" if x==1 else "No")
    monthly_data_usage_gb     = st.number_input("Monthly Data Usage (GB)",   min_value=0.0, max_value=100.0, value=8.0, step=0.5)
    recent_app_installs       = st.number_input("Recent App Installs (30d)", min_value=0,   max_value=30,   value=2)
 
    st.markdown("---")
    st.markdown("<div class='section-label'>Financial Profile</div>", unsafe_allow_html=True)
    existing_debt_amount  = st.number_input("Existing Debt Amount (₦)",  min_value=0, max_value=10000000, value=50000,  step=5000)
    loan_amount_requested = st.number_input("Loan Amount Requested (₦)", min_value=0, max_value=10000000, value=200000, step=10000)
    savings_worth         = st.number_input("Savings Worth (₦)",         min_value=0, max_value=10000000, value=80000,  step=5000)
 
    st.markdown("---")
    run_btn = st.button("▶  RUN CREDIT ASSESSMENT")
 
# ── Main Panel ────────────────────────────────────────────────────────────────
if not run_btn:
    st.markdown("""
    <div style='text-align:center; padding: 80px 40px; color: #5a7a99;'>
        <div style='font-size:48px; opacity:0.3;'>◈</div>
        <div style='font-family:monospace; font-size:13px; margin-top:16px;'>
            Enter borrower data in the sidebar and click <strong style='color:#00c896;'>RUN CREDIT ASSESSMENT</strong> to generate a credit risk profile.
        </div>
    </div>
    """, unsafe_allow_html=True)
 
else:
    # Build input dataframe
    input_data = pd.DataFrame([{
        'unique_apps_per_day':        unique_apps_per_day,
        'avg_daily_screen_time_hrs':  avg_daily_screen_time_hrs,
        'financial_apps_installed':   financial_apps_installed,
        'online_txn_count_last_30d':  online_txn_count_last_30d,
        'avg_txn_amount':             avg_txn_amount,
        'bank_sms_count':             bank_sms_count,
        'avg_distance_travelled_km':  avg_distance_travelled_km,
        'places_visited_weekly':      places_visited_weekly,
        'calls_per_day':              calls_per_day,
        'distinct_contacts_weekly':   distinct_contacts_weekly,
        'avg_call_duration_mins':     avg_call_duration_mins,
        'social_media_pct':           social_media_pct,
        'finance_app_time_pct':       finance_app_time_pct,
        'regular_sleep_pattern':      regular_sleep_pattern,
        'battery_charging_regular':   battery_charging_regular,
        'monthly_data_usage_gb':      monthly_data_usage_gb,
        'recent_app_installs':        recent_app_installs,
        'existing_debt_amount':       existing_debt_amount,
        'loan_amount_requested':      loan_amount_requested,
        'savings_worth':              savings_worth,
    }])
 
    # Scale
    try:
        input_scaled = scaler.transform(input_data)
    except Exception:
        feature_names = scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else input_data.columns
        input_aligned = input_data.reindex(columns=feature_names, fill_value=0)
        input_scaled  = scaler.transform(input_aligned)
 
    # Predict
    probability  = model.predict_proba(input_scaled)[0]
    default_prob = probability[1] * 100
    repaid_prob  = probability[0] * 100
 
    # Risk classification
    if default_prob < 40:
        risk_class = "low";    risk_label = "LOW RISK";    risk_sub = "Borrower demonstrates strong repayment indicators."
    elif default_prob < 46:
        risk_class = "medium"; risk_label = "MEDIUM RISK"; risk_sub = "Moderate indicators — manual review recommended."
    else:
        risk_class = "high";   risk_label = "HIGH RISK";   risk_sub = "Multiple adverse indicators detected."
 
    # Log to Firebase
    log_assessment(db, borrower_id, risk_label, default_prob, input_data)
 
    # Show Firebase error if any
    if "firebase_error" in st.session_state:
        st.error(f"🔥 Firebase log error: {st.session_state['firebase_error']}")
        del st.session_state["firebase_error"]
 
    # Row 1: Verdict + Metrics
    col_v, col_m1, col_m2, col_m3 = st.columns([2, 1, 1, 1])
    with col_v:
        title_class = f"verdict-title-{'low' if risk_class=='low' else 'high' if risk_class=='high' else 'med'}"
        st.markdown(f"""
        <div class='verdict-{risk_class}'>
            <div class='{title_class}'>{risk_label}</div>
            <div class='verdict-sub'>{risk_sub}</div>
            <div style='margin-top:12px; font-family:monospace; font-size:11px; color:#5a7a99;'>BORROWER: {borrower_id}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m1: st.metric("Default Probability",   f"{default_prob:.1f}%")
    with col_m2: st.metric("Repayment Probability", f"{repaid_prob:.1f}%")
    with col_m3: st.metric("Model Confidence",      f"{max(default_prob, repaid_prob):.1f}%")
 
    st.markdown("---")
 
    # Row 2: Summary + Feature Importance
    col_profile, col_feat = st.columns([1, 1])
    with col_profile:
        st.markdown("#### 📊 Borrower Summary")
        summary_data = {
            "Field": ["Borrower ID","Loan Requested","Existing Debt","Savings Worth","Online Txns (30d)","Finance App Time","Unique Apps/Day","Monthly Data (GB)"],
            "Value": [borrower_id, f"₦{loan_amount_requested:,}", f"₦{existing_debt_amount:,}", f"₦{savings_worth:,}", str(online_txn_count_last_30d), f"{finance_app_time_pct:.1f}%", str(unique_apps_per_day), f"{monthly_data_usage_gb:.1f} GB"]
        }
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
 
    with col_feat:
        st.markdown("#### 🔍 Feature Importance (Random Forest)")
        importances   = model.feature_importances_
        feature_names = list(scaler.feature_names_in_) if hasattr(scaler, 'feature_names_in_') else input_data.columns.tolist()
        n = min(len(feature_names), len(importances))
        feat_df = pd.DataFrame({'Feature': feature_names[:n], 'Importance': importances[:n]}).sort_values('Importance', ascending=True).tail(10)
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('#0b0f14')
        ax.set_facecolor('#121920')
        colors = ['#ff4d6d' if f in ['existing_debt_amount','loan_amount_requested'] else '#00c896' for f in feat_df['Feature']]
        ax.barh(feat_df['Feature'], feat_df['Importance'], color=colors, edgecolor='none', height=0.6)
        ax.set_xlabel('Importance Score', color='#8ba3bd', fontsize=9)
        ax.tick_params(colors='#8ba3bd', labelsize=8)
        for spine in ['top','right']: ax.spines[spine].set_visible(False)
        for spine in ['left','bottom']: ax.spines[spine].set_color('#1e2d3d')
        ax.grid(axis='x', alpha=0.2, color='#1e2d3d')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
 
    st.markdown("---")
 
    # Row 3: Risk Dimensions
    st.markdown("#### 📈 Risk Dimension Breakdown")
    d1, d2, d3, d4 = st.columns(4)
    digital_score  = min(100, (online_txn_count_last_30d*2 + finance_app_time_pct + unique_apps_per_day*3))
    financial_score= max(0, 100 - (existing_debt_amount / max(savings_worth, 1)) * 20)
    behaviour_score= (regular_sleep_pattern*20 + battery_charging_regular*20 + min(calls_per_day*3,30) + min(distinct_contacts_weekly*0.5,30))
    exposure_score = max(0, 100 - (loan_amount_requested / max(savings_worth + avg_txn_amount*12, 1)) * 40)
    for col, label, score in zip([d1,d2,d3,d4], ["Digital Engagement","Financial Health","Behavioural Consistency","Loan Exposure"], [digital_score,financial_score,behaviour_score,exposure_score]):
        with col: st.metric(label, f"{min(100,max(0,score)):.0f}/100")
 
    st.markdown("---")
 
    # Row 4: Decision
    st.markdown("#### 📝 Assessment Decision")
    if risk_class == "low":
        st.success(f"✅ **RECOMMENDATION: APPROVE** — {borrower_id} presents a low default risk of {default_prob:.1f}%. Digital behavioural indicators suggest consistent financial engagement and strong repayment capacity.")
    elif risk_class == "medium":
        st.warning(f"⚠️ **RECOMMENDATION: MANUAL REVIEW** — {borrower_id} presents a moderate default risk of {default_prob:.1f}%. Additional documentation or collateral review is advised before approval.")
    else:
        st.error(f"❌ **RECOMMENDATION: DECLINE** — {borrower_id} presents a high default risk of {default_prob:.1f}%. Multiple adverse behavioural indicators detected. Loan application not recommended at this time.")
 








