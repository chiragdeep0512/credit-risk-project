# ============================================================
#  CREDIT RISK SCORING — Home Credit 2023
#  Phase 06: Streamlit App
#  Author: Naman Deep Singh
#  GitHub: github.com/chiragdeep0512
# ============================================================
#
#  INSTALL:
#  pip install streamlit plotly lightgbm shap pandas numpy
#
#  RUN (in PyCharm Terminal):
#  streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import lightgbm as lgb
import json
from pathlib import Path

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Scoring",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size:2rem; font-weight:700; color:#1a237e;
        text-align:center; padding:1rem 0 0.3rem 0;
    }
    .sub-header {
        font-size:1rem; color:#546e7a;
        text-align:center; margin-bottom:1.5rem;
    }
    .kpi-card {
        background:#f8f9fa; border-radius:12px;
        padding:1.2rem; text-align:center;
        border:1px solid #e0e0e0;
    }
    .kpi-val   { font-size:1.8rem; font-weight:700; color:#1a237e; }
    .kpi-label { font-size:0.8rem; color:#78909c; margin-top:0.2rem; }
    .green-box  { background:#e8f5e9; border-left:5px solid #2e7d32;
                  padding:1rem; border-radius:8px; margin:0.5rem 0; }
    .yellow-box { background:#fff8e1; border-left:5px solid #f9a825;
                  padding:1rem; border-radius:8px; margin:0.5rem 0; }
    .red-box    { background:#ffebee; border-left:5px solid #c62828;
                  padding:1rem; border-radius:8px; margin:0.5rem 0; }
    .insight    { background:#e3f2fd; border-left:4px solid #1565c0;
                  padding:0.7rem 1rem; border-radius:6px;
                  font-size:0.9rem; margin:0.4rem 0; }
    .stButton>button {
        background:#1a237e; color:white; border-radius:8px;
        border:none; padding:0.6rem 2rem; font-size:1rem;
        font-weight:600; width:100%;
    }
</style>
""", unsafe_allow_html=True)

# ── PATHS ──────────────────────────────────────────────────────
BASE   = Path(__file__).parent
MODELS = BASE / 'models'
CHARTS = BASE / 'data' / 'processed' / 'model_charts'


# ── CACHED LOADERS ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    m = lgb.Booster(model_file=str(MODELS / 'best_model_lgb.txt'))
    with open(MODELS / 'best_threshold.json') as f:
        info = json.load(f)
    return m, info

@st.cache_data
def load_importance():
    return pd.read_csv(MODELS / 'feature_importance.csv')

@st.cache_data
def load_shap_imp():
    return pd.read_csv(MODELS / 'shap_feature_importance.csv')


# ── SIDEBAR ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏦 Credit Risk System")
    st.markdown("**Home Credit 2023**")
    st.markdown("---")
    page = st.radio("Navigate", [
        "🏠 Overview",
        "🔍 Risk Analyzer",
        "📊 SHAP",
        "📈 Insights"
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**Stack:** LightGBM · SHAP · Streamlit")
    st.markdown("**Author:** Naman Deep Singh")
    st.markdown("[GitHub](https://github.com/chiragdeep0512) · "
                "[LinkedIn](https://linkedin.com/in/naman-deep-singh-nds05)")


# ============================================================
# PAGE 1 — OVERVIEW
# ============================================================

if page == "🏠 Overview":

    st.markdown('<div class="main-header">🏦 Credit Risk Scoring System</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Home Credit 2023 · LightGBM + SHAP Explainability</div>',
                unsafe_allow_html=True)

    try:
        model, info = load_model()
        auc   = info.get('auc', 'N/A')
        f1    = info.get('f1',  'N/A')
        pr    = info.get('precision', 'N/A')
        rc    = info.get('recall',    'N/A')
        thr   = info.get('threshold', 'N/A')
        nf    = len(info.get('feature_cols', []))
    except Exception as e:
        st.error(f"Could not load model: {e}")
        st.stop()

    # KPI row
    st.markdown("### 📊 Model Performance")
    c1,c2,c3,c4,c5 = st.columns(5)
    for col, val, lbl in zip(
        [c1,c2,c3,c4,c5],
        [auc,f1,pr,rc,nf],
        ["AUC-ROC","F1 Score","Precision","Recall","Features"]
    ):
        col.markdown(
            f'<div class="kpi-card"><div class="kpi-val">{val}</div>'
            f'<div class="kpi-label">{lbl}</div></div>',
            unsafe_allow_html=True
        )

    st.markdown("---")

    # Charts row 1
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 📈 ROC Curve")
        p = CHARTS / '01_roc_curve.png'
        if p.exists(): st.image(str(p), use_column_width=True)
    with col2:
        st.markdown("#### 🎯 Feature Importance")
        p = CHARTS / '03_feature_importance.png'
        if p.exists(): st.image(str(p), use_column_width=True)

    st.markdown("---")

    # Charts row 2
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("#### 🔲 Confusion Matrix")
        p = CHARTS / '02_confusion_matrix.png'
        if p.exists(): st.image(str(p), use_column_width=True)
    with col4:
        st.markdown("#### 📉 Stability (Gini/Week)")
        p = CHARTS / '05_stability.png'
        if p.exists(): st.image(str(p), use_column_width=True)

    st.markdown("---")
    st.markdown("### 📋 Project Summary")
    col5, col6 = st.columns(2)
    with col5:
        st.markdown("""
        **Dataset**
        - Source: Home Credit 2023 (Kaggle)
        - Training cases: 1,526,659
        - Features engineered: 414
        - Default rate: 3.14%
        - Time-based train/val split (WEEK_NUM)

        **Model**
        - Algorithm: LightGBM (GBDT)
        - Imbalance: is_unbalance = True
        - Early stopping: 30 rounds
        """)
    with col6:
        st.markdown("""
        **Feature Engineering**
        - Depth=0,1,2 table aggregations
        - Bureau DPD & amount features
        - Previous application history
        - Tax registry aggregations
        - Date & ratio features

        **Explainability**
        - SHAP TreeExplainer
        - Global summary & bar plots
        - Local waterfall per applicant
        """)


# ============================================================
# PAGE 2 — RISK ANALYZER
# ============================================================

elif page == "🔍 Risk Analyzer":

    st.markdown('<div class="main-header">🔍 Applicant Risk Analyzer</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Enter applicant details to predict default probability</div>',
                unsafe_allow_html=True)

    try:
        model, info = load_model()
        feat_cols = info.get('feature_cols', [])
        threshold = info.get('threshold', 0.5)
    except:
        st.error("Model not loaded. Run Phase 04 first.")
        st.stop()

    with st.form("risk_form"):
        st.markdown("### 📝 Applicant Details")
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**💰 Credit Info**")
            cred_amt  = st.number_input("Credit Amount (₹)", 0, 10000000, 200000, 10000)
            annuity   = st.number_input("Monthly Annuity (₹)", 0, 500000, 8000, 500)
            curr_debt = st.number_input("Current Debt (₹)", 0, 5000000, 50000, 5000)

        with c2:
            st.markdown("**📋 Application History**")
            prev_apps = st.number_input("Previous Applications", 0, 50, 2)
            avg_amt   = st.number_input("Avg Prev Credit Amount (₹)", 0, 5000000, 150000, 10000)
            avg_ann   = st.number_input("Avg Prev Annuity (₹)", 0, 200000, 6000, 500)

        with c3:
            st.markdown("**🏦 Bureau Info**")
            bur_a    = st.number_input("Bureau A Records", 0, 100, 3)
            bur_b    = st.number_input("Bureau B Records", 0, 100, 2)
            avg_dpd  = st.number_input("Avg Days Past Due", 0.0, 365.0, 0.0, 0.5)

        c4, c5 = st.columns(2)
        with c4:
            st.markdown("**👤 Personal**")
            persons   = st.number_input("Persons in Application", 1, 10, 1)
            cancel_r  = st.number_input("Unique Cancellation Reasons", 0, 10, 0)
            tax_b     = st.number_input("Tax Registry B Records", 0, 50, 1)
        with c5:
            st.markdown("**📅 Date**")
            app_month = st.selectbox("Month", list(range(1,13)), index=0)
            app_year  = st.selectbox("Year", [2019,2020], index=0)
            app_dow   = st.selectbox("Day of Week",
                        ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"])

        submitted = st.form_submit_button("🔍 Analyze Risk")

    if submitted:
        dow_map = {"Mon":0,"Tue":1,"Wed":2,"Thu":3,"Fri":4,"Sat":5,"Sun":6}
        inp = {
            'ap1_count':                           float(prev_apps),
            'ap1_credamount_590A_mean':            float(avg_amt),
            'ap1_annuity_853A_mean':               float(avg_ann),
            'ap1_credamount_590A_max':             float(avg_amt*1.3),
            'ap1_annuity_853A_max':                float(avg_ann*1.2),
            'ap1_credacc_actualbalance_314A_mean': float(curr_debt*0.5),
            'ap1_actualdpd_943P_mean':             float(avg_dpd),
            'ap1_cancelreason_3545846M_nunique':   float(cancel_r),
            's0_credamount_770A':                  float(cred_amt),
            's0_annuity_780A':                     float(annuity),
            's0_currdebt_22A':                     float(curr_debt),
            'cb0_pmts_dpd_1073P':                  float(avg_dpd),
            'cb0_pmts_overdue_1140A':              float(curr_debt*0.1),
            'cb0_dpdmaxdateyear_596T':             float(app_year),
            'bura1_count':                         float(bur_a),
            'bura1_annuity_853A_mean':             float(avg_ann),
            'bura1_annuity_853A_max':              float(avg_ann*1.5),
            'burb1_count':                         float(bur_b),
            'burb1_pmts_dpd_1073P_mean':           float(avg_dpd),
            'burb1_pmts_dpd_1073P_max':            float(avg_dpd*2),
            'burb1_pmts_overdue_1140A_mean':       float(curr_debt*0.05),
            'burb1_pmts_overdue_1140A_sum':        float(curr_debt*0.15),
            'per1_total_persons':                  float(persons),
            'taxa_count':                          1.0,
            'taxb_count':                          float(tax_b),
            'fe_month':                            float(app_month),
            'fe_year':                             float(app_year),
            'fe_dayofweek':                        float(dow_map[app_dow]),
            'WEEK_NUM':                            50.0,
        }

        vec  = np.array([[inp.get(f, 0.0) for f in feat_cols]], dtype=np.float32)
        prob = float(model.predict(vec)[0])

        if prob < 0.20:
            band, color, emoji, advice, css = (
                "LOW RISK", "#2e7d32", "✅",
                "Strong credit profile. Recommended for approval.", "green-box"
            )
        elif prob < 0.50:
            band, color, emoji, advice, css = (
                "MEDIUM RISK", "#f9a825", "⚠️",
                "Moderate risk. Consider additional verification.", "yellow-box"
            )
        else:
            band, color, emoji, advice, css = (
                "HIGH RISK", "#c62828", "❌",
                "High default risk. Recommend rejection or collateral.", "red-box"
            )

        st.markdown("---")
        st.markdown("### 🎯 Result")

        cg, cr = st.columns([1,1])
        with cg:
            fig = go.Figure(go.Indicator(
                mode  = "gauge+number",
                value = round(prob*100, 2),
                title = {'text': "Default Probability (%)"},
                gauge = {
                    'axis':  {'range': [0,100]},
                    'bar':   {'color': color},
                    'steps': [
                        {'range':[0,20],  'color':'#e8f5e9'},
                        {'range':[20,50], 'color':'#fff8e1'},
                        {'range':[50,100],'color':'#ffebee'},
                    ],
                    'threshold': {
                        'line': {'color':'black','width':4},
                        'thickness':0.85,
                        'value': threshold*100
                    }
                },
                number={'suffix':'%','font':{'size':36}}
            ))
            fig.update_layout(height=280, margin=dict(t=50,b=10,l=20,r=20))
            st.plotly_chart(fig, use_container_width=True)

        with cr:
            st.markdown(
                f'<div class="{css}"><h2>{emoji} {band}</h2>'
                f'<p><b>Probability: {prob*100:.2f}%</b></p>'
                f'<p>{advice}</p></div>',
                unsafe_allow_html=True
            )
            decision = "✅ APPROVE" if prob < threshold else "❌ REJECT"
            st.metric("Decision", decision,
                      f"{prob*100:.1f}% vs {threshold*100:.1f}% threshold")

            st.markdown("**⚠️ Risk Signals**")
            risks = []
            if avg_dpd > 10:   risks.append(f"High DPD: {avg_dpd:.0f} days")
            if cancel_r > 2:   risks.append("Multiple cancellations")
            if bur_a == 0:     risks.append("No bureau A records")
            if prev_apps > 10: risks.append("Too many previous applications")
            if curr_debt > cred_amt*0.5: risks.append("High debt vs loan amount")

            for r in (risks or ["No major risk signals"]):
                icon = "⚠️" if risks else "✅"
                st.markdown(f'<div class="insight">{icon} {r}</div>',
                            unsafe_allow_html=True)


# ============================================================
# PAGE 3 — SHAP
# ============================================================

elif page == "📊 SHAP":

    st.markdown('<div class="main-header">📊 SHAP Explainability</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Why does the model predict what it predicts?</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="insight">
    <b>SHAP</b> (SHapley Additive exPlanations): each feature gets a score showing
    how much it <b>increased</b> (red/positive) or <b>decreased</b> (blue/negative)
    the default probability for each applicant.
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🌍 Global — All Applicants")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Beeswarm Summary**")
        p = CHARTS/'06_shap_summary.png'
        if p.exists(): st.image(str(p), use_column_width=True)
        st.caption("Each dot = one applicant. Red = high feature value.")
    with c2:
        st.markdown("**Mean |SHAP| Bar**")
        p = CHARTS/'07_shap_bar.png'
        if p.exists(): st.image(str(p), use_column_width=True)
        st.caption("Longer bar = more globally important feature.")

    st.markdown("---")
    st.markdown("### 🔍 Local — Single Applicant Explained")
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Defaulter Waterfall**")
        p = CHARTS/'08_shap_waterfall_default.png'
        if p.exists(): st.image(str(p), use_column_width=True)
        st.caption("Why this applicant was predicted HIGH risk.")
    with c4:
        st.markdown("**Non-Defaulter Waterfall**")
        p = CHARTS/'09_shap_waterfall_nondefault.png'
        if p.exists(): st.image(str(p), use_column_width=True)
        st.caption("Why this applicant was predicted LOW risk.")

    st.markdown("---")
    st.markdown("### 📈 Feature Dependence")
    p = CHARTS/'10_shap_dependence.png'
    if p.exists():
        st.image(str(p), use_column_width=True)
        st.caption("How the top feature value affects default probability.")

    st.markdown("---")
    st.markdown("### 📋 SHAP Feature Table")
    try:
        si = load_shap_imp()
        st.dataframe(
            si[['Feature','Mean_SHAP','Business_Reason']].head(20),
            use_container_width=True
        )
    except:
        st.info("Run Phase 05 to generate SHAP importance file.")

    st.markdown("---")
    st.markdown("### 💡 Business Insights")
    for title, text in [
        ("🔴 DPD History is #1 predictor",
         "Applicants with high days-past-due in bureau have 3-4x higher default risk."),
        ("🟡 Previous application count matters",
         "Too many applications signals financial stress — a key risk flag."),
        ("🟢 Thin credit file = higher risk",
         "Zero bureau records = unknown risk. Models treat this conservatively."),
        ("🔵 Credit amount vs annuity ratio",
         "High loan amount relative to historical annuity = over-leveraging risk."),
    ]:
        st.markdown(f'<div class="insight" style="color:#000000;"><b>{title}</b><br>{text}</div>',
                    unsafe_allow_html=True)


# ============================================================
# PAGE 4 — INSIGHTS
# ============================================================

elif page == "📈 Insights":

    st.markdown('<div class="main-header">📈 Model Insights</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Feature importance, score distribution, model config</div>',
                unsafe_allow_html=True)

    try:
        model, info = load_model()
        feat_cols = info.get('feature_cols', [])
    except:
        st.error("Model not found.")
        st.stop()

    # Score distribution
    st.markdown("### 📊 Score Distribution")
    p = CHARTS/'04_score_distribution.png'
    if p.exists():
        st.image(str(p), use_column_width=True)
        st.caption("Ideal: defaulters (red) at high probability, "
                   "non-defaulters (blue) at low probability.")

    st.markdown("---")

    # Interactive feature importance
    st.markdown("### 🎯 Interactive Feature Importance")
    try:
        imp = load_importance().head(20)
        fig = px.bar(
            imp.iloc[::-1], x='Importance', y='Feature',
            orientation='h', color='Importance',
            color_continuous_scale='Blues',
            title='Top 20 Features by Gain (LightGBM)'
        )
        fig.update_layout(height=480, showlegend=False,
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    except:
        st.info("Feature importance CSV not found.")

    st.markdown("---")

    # Risk simulator
    st.markdown("### 🎲 Quick Risk Simulator")
    c1, c2 = st.columns(2)
    with c1:
        s_dpd  = st.slider("Days Past Due",    0, 180, 0)
        s_loan = st.slider("Loan Amount (₹)", 50000, 1000000, 200000, 10000)
    with c2:
        s_prev = st.slider("Previous Apps",    0, 20, 2)
        s_debt = st.slider("Current Debt (₹)", 0, 500000, 50000, 10000)

    sim = min(0.05
              + min(s_dpd/30*0.15, 0.4)
              + min(s_loan/500000*0.1, 0.2)
              + min(s_prev/20*0.1, 0.15)
              + min(s_debt/300000*0.1, 0.2), 0.95)

    band_s = ("LOW RISK ✅" if sim < 0.20
              else "MEDIUM RISK ⚠️" if sim < 0.50
              else "HIGH RISK ❌")
    st.metric("Simulated Default Probability", f"{sim*100:.1f}%", band_s)

    st.markdown("---")

    # Model config table
    st.markdown("### ⚙️ Model Configuration")
    st.table(pd.DataFrame([
        ("Algorithm",          "LightGBM GBDT"),
        ("Objective",          "Binary Classification"),
        ("Imbalance handling", "is_unbalance = True"),
        ("Early stopping",     "30 rounds"),
        ("Features used",      len(feat_cols)),
        ("Train strategy",     "Time-based split (WEEK_NUM ≤ 72)"),
        ("AUC-ROC",            info.get('auc','N/A')),
        ("F1 Score",           info.get('f1', 'N/A')),
        ("Decision threshold", f"{info.get('threshold',0):.4f}"),
    ], columns=["Parameter","Value"]))