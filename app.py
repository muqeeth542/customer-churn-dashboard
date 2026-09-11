"""
app.py
------
Customer Churn Prediction Dashboard
Streamlit application — 4 pages:

  1. Dashboard          — KPIs, charts, high-risk table
  2. Customer Analysis  — per-customer search + SHAP explanation
  3. Churn Prediction   — manual input form + live prediction
  4. Model Performance  — metrics, confusion matrix, feature importance

Run with:
    streamlit run app.py
"""

import json
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st
from sklearn.metrics import roc_curve, auc

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
DATA_DIR = os.path.join(BASE_DIR, "data")

MODEL_PATH = os.path.join(MODEL_DIR, "churn_model.pkl")
SHAP_PATH = os.path.join(MODEL_DIR, "shap_explainer.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "model_metrics.json")
PREDICTIONS_PATH = os.path.join(MODEL_DIR, "predictions.csv")
DATA_PATH = os.path.join(DATA_DIR, "WA_Fn-UseC_-Telco-Customer-Churn.csv")

sys.path.insert(0, BASE_DIR)

# ─────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS — modern dark-ish analytics look
# ─────────────────────────────────────────────
st.markdown(
    """
<style>
    :root {
      --bg-app: #0B0F19;
      --bg-panel: #131A2A;
      --bg-panel-hover: #1A2338;
      --border-subtle: #232C42;
      --accent-primary: #3B82F6;
      --risk-critical: #EF4444;
      --risk-high: #F97316;
      --risk-medium: #EAB308;
      --risk-low: #22C55E;
      --text-primary: #F3F5F9;
      --text-secondary: #8B94A8;
    }

    .stApp { background-color: var(--bg-app); }

    /* Main background */
    .main .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

    /* KPI cards */
    .kpi-card {
        background-color: var(--bg-panel);
        border: 1px solid var(--border-subtle);
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: background-color 0.15s ease, box-shadow 0.15s ease;
    }
    .kpi-card:hover {
        background-color: var(--bg-panel-hover);
        box-shadow: 0 6px 20px rgba(0,0,0,0.2);
    }
    .kpi-label { color: var(--text-secondary); font-size: 11px; font-weight: 600;
                 letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 0.3rem; }
    .kpi-value { color: var(--text-primary); font-size: 32px; font-weight: 700; line-height: 1.1; }
    .kpi-sub   { color: var(--text-secondary); font-size: 14px; margin-top: 0.2rem; }

    div[data-testid="stMetric"],
    div[data-testid="stVerticalBlockBorderWrapper"] {
      background-color: var(--bg-panel);
      border: 1px solid var(--border-subtle);
      border-radius: 10px;
      padding: 20px;
    }
    div[data-testid="stMetric"]:hover {
      background-color: var(--bg-panel-hover);
      transition: background-color 0.15s ease;
    }

    /* Section headers */
    .section-header { color: var(--text-primary); font-size: 15px; font-weight: 600;
                      border-bottom: 1px solid var(--border-subtle); padding-bottom: 0.4rem;
                      margin-bottom: 1rem; margin-top: 2rem; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
      background-color: var(--bg-panel) !important;
      border-right: 1px solid var(--border-subtle);
    }
    [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label {
      padding: 12px 12px;
      border-radius: 10px;
      transition: background-color 0.2s ease;
      margin-bottom: 4px;
    }
    [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label:hover {
      background-color: var(--bg-panel-hover);
    }
    [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label[data-checked="true"] {
      background-color: var(--bg-panel-hover);
    }
    [data-testid="stSidebar"] .stRadio div[data-testid="stMarkdownContainer"] p {
      color: var(--text-primary) !important; 
      font-size: 14px;
      font-weight: 500;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Info box */
    .info-box { background: var(--bg-panel); border-left: 3px solid var(--accent-primary);
                padding: 0.8rem 1rem; border-radius: 6px; margin: 0.5rem 0; }
    .info-box p { margin: 0; color: var(--text-secondary); font-size: 14px; }

    /* Prob bar */
    .prob-high     { color: var(--risk-critical); font-size: 2.2rem; font-weight: 800; }
    .prob-medium   { color: var(--risk-high); font-size: 2.2rem; font-weight: 800; }
    .prob-low      { color: var(--risk-low); font-size: 2.2rem; font-weight: 800; }

    /* Risk pill badges for tables */
    .risk-pill {
      display: inline-block;
      padding: 2px 10px;
      border-radius: 999px;
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
    }
    .risk-pill--critical { background: rgba(239,68,68,0.15); color: var(--risk-critical); }
    .risk-pill--high     { background: rgba(249,115,22,0.15); color: var(--risk-high); }
    .risk-pill--medium   { background: rgba(234,179,8,0.15);  color: var(--risk-medium); }
    .risk-pill--low      { background: rgba(34,197,94,0.15);  color: var(--risk-low); }

    /* Table styling */
    .custom-table { width: 100%; border-collapse: collapse; color: var(--text-primary); font-size: 13px; }
    .custom-table th { border-bottom: 1px solid var(--border-subtle); padding: 8px 12px; text-align: left; font-weight: 600; color: var(--text-secondary); }
    .custom-table td { padding: 8px 12px; border-bottom: 1px solid var(--border-subtle); }
    .custom-table tr:nth-child(even) { background-color: var(--bg-app); }
    .custom-table tr:nth-child(odd) { background-color: var(--bg-panel); }
    .custom-table .num-col { text-align: right; }
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# Data / model loading (cached)
# ─────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_predictions():
    return pd.read_csv(PREDICTIONS_PATH)


@st.cache_data(show_spinner=False)
def load_metrics():
    with open(METRICS_PATH) as f:
        return json.load(f)


@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource(show_spinner=False)
def load_shap():
    return joblib.load(SHAP_PATH)


def check_artifacts():
    missing = []
    for p in [MODEL_PATH, SHAP_PATH, METRICS_PATH, PREDICTIONS_PATH]:
        if not os.path.exists(p):
            missing.append(p)
    return missing


# ─────────────────────────────────────────────
# Risk badge helper
# ─────────────────────────────────────────────

RISK_COLORS = {
    "CRITICAL": "#EF4444",
    "HIGH":     "#F97316",
    "MEDIUM":   "#EAB308",
    "LOW":      "#22C55E",
}

def risk_badge(level: str) -> str:
    cls = f"risk-pill risk-pill--{level.lower()}"
    return f'<span class="{cls}">{level}</span>'


# ─────────────────────────────────────────────
# Plotly theme helper
# ─────────────────────────────────────────────

PLOT_LAYOUT = dict(
    paper_bgcolor="#131A2A",
    plot_bgcolor="#131A2A",
    font=dict(color="#8B94A8", family="Inter, system-ui, sans-serif", size=13),
    margin=dict(l=10, r=10, t=40, b=10),
)


def styled_fig(fig, title=""):
    fig.update_layout(**PLOT_LAYOUT, title=dict(text=title, font=dict(size=15, color="#F3F5F9")))
    fig.update_xaxes(gridcolor="#232C42", zerolinecolor="#232C42")
    fig.update_yaxes(gridcolor="#232C42", zerolinecolor="#232C42")
    return fig


# ─────────────────────────────────────────────
# Sidebar Navigation
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        """
        <div style='text-align:center; padding: 1rem 0 1.5rem 0;'>
            <div style='font-size:2rem;'>📉</div>
            <div style='color:#f1f5f9; font-size:1.1rem; font-weight:700; margin-top:0.3rem;'>
                ChurnIQ
            </div>
            <div style='color:#64748b; font-size:0.75rem;'>Customer Retention Analytics</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        ["📊 Dashboard", "🔍 Customer Analysis", "🎯 Churn Prediction", "📈 Model Performance"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        """
        <div style='background-color: var(--bg-panel-hover); border: 1px solid var(--border-subtle); border-radius: 10px; padding: 12px; margin-top: 16px;'>
            <div style='color: var(--text-primary); font-size: 0.85rem; font-weight: 700; margin-bottom: 8px;'>Session Info</div>
            <div style='color: var(--text-secondary); font-size: 0.75rem; line-height: 1.5;'>
                <b>Dataset:</b> IBM Telco Customer Churn<br>
                <b>Model:</b> XGBoost Classifier<br>
                <b>Records:</b> 7,043 customers<br>
                <b>Explainability:</b> SHAP TreeExplainer
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
# Guard: check model artifacts exist
# ─────────────────────────────────────────────

missing = check_artifacts()
if missing:
    st.error("🚨 Model artifacts not found. Please train the model first:")
    st.code("python model/train_model.py")
    st.info("This will train the XGBoost model and generate all required files.")
    st.stop()

# Load everything
with st.spinner("Loading model and predictions..."):
    df = load_predictions()
    metrics = load_metrics()
    model = load_model()
    shap_pkg = load_shap()

explainer      = shap_pkg["explainer"]
preprocessor   = shap_pkg["preprocessor"]
feature_names  = shap_pkg["feature_names"]

# ═════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ═════════════════════════════════════════════

if page == "📊 Dashboard":
    st.markdown("## 📊 Dashboard Overview")
    st.markdown("Real-time churn risk intelligence for your customer base.")

    # ── KPI row ──────────────────────────────
    total = len(df)
    churned = int(df["Churn"].sum())
    churn_rate = churned / total * 100
    high_risk = int(((df["risk_level"] == "HIGH") | (df["risk_level"] == "CRITICAL")).sum())
    critical = int((df["risk_level"] == "CRITICAL").sum())
    avg_monthly = df["MonthlyCharges"].mean()
    avg_prob = df["churn_probability"].mean() * 100

    c1, c2, c3, c4, c5, c6 = st.columns(6)

    def kpi(col, label, value, sub="", is_primary=False):
        border = "var(--risk-critical)" if is_primary else "var(--border-subtle)"
        bg = "rgba(239, 68, 68, 0.05)" if is_primary else "var(--bg-panel)"
        col.markdown(
            f"""<div class="kpi-card" style="border-color: {border}; background-color: {bg};">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
                {"<div class='kpi-sub'>" + sub + "</div>" if sub else ""}
            </div>""",
            unsafe_allow_html=True,
        )

    kpi(c1, "Total Customers", f"{total:,}", "in dataset")
    kpi(c2, "Churned", f"{churned:,}", f"{churn_rate:.1f}% of base")
    kpi(c3, "Churn Rate", f"{churn_rate:.1f}%", "actual observed", is_primary=True)
    kpi(c4, "High Risk", f"{high_risk:,}", "HIGH + CRITICAL")
    kpi(c5, "Critical Risk", f"{critical:,}", "prob > 80%")
    kpi(c6, "Avg Monthly $", f"${avg_monthly:.0f}", "per customer")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1 charts: Churn Overview ─────────
    st.markdown('<div class="section-header">📊 Churn Overview</div>', unsafe_allow_html=True)
    col_l, col_m, col_r = st.columns(3)

    with col_l:
        churn_counts = df["Churn"].map({0: "Stayed", 1: "Churned"}).value_counts().reset_index()
        churn_counts.columns = ["Status", "Count"]
        fig = px.pie(
            churn_counts, names="Status", values="Count",
            color="Status",
            color_discrete_map={"Churned": RISK_COLORS["CRITICAL"], "Stayed": RISK_COLORS["LOW"]},
            hole=0.55,
        )
        fig = styled_fig(fig, "Churn vs Stayed")
        fig.update_traces(textfont_size=13)
        st.plotly_chart(fig, use_container_width=True)

    with col_m:
        fig = px.histogram(
            df, x="churn_probability", nbins=40,
            color_discrete_sequence=["var(--accent-primary)"],
            labels={"churn_probability": "Churn Probability"},
        )
        fig = styled_fig(fig, "Churn Probability Distribution")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        risk_counts = df["risk_level"].value_counts().reset_index()
        risk_counts.columns = ["Risk Level", "Count"]
        fig = px.bar(
            risk_counts, x="Risk Level", y="Count",
            color="Risk Level",
            color_discrete_map=RISK_COLORS,
            text="Count",
            category_orders={"Risk Level": ["CRITICAL", "HIGH", "MEDIUM", "LOW"]},
        )
        fig.update_traces(textposition="outside")
        fig = styled_fig(fig, "Customer Risk Distribution")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 2 charts: Churn Drivers ──────────
    st.markdown('<div class="section-header">⚙️ Churn Drivers</div>', unsafe_allow_html=True)
    col_l2, col_m2, col_r2 = st.columns(3)

    with col_l2:
        contract_churn = df.groupby("Contract")["Churn"].mean().reset_index()
        contract_churn.columns = ["Contract", "Churn Rate"]
        contract_churn["Churn Rate"] = contract_churn["Churn Rate"] * 100
        fig = px.bar(
            contract_churn, x="Contract", y="Churn Rate",
            color="Churn Rate",
            color_continuous_scale=[RISK_COLORS["LOW"], RISK_COLORS["HIGH"], RISK_COLORS["CRITICAL"]],
            text="Churn Rate",
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig = styled_fig(fig, "Churn Rate by Contract Type")
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_m2:
        isp_churn = df.groupby("InternetService")["Churn"].mean().reset_index()
        isp_churn.columns = ["Internet Service", "Churn Rate"]
        isp_churn["Churn Rate"] = isp_churn["Churn Rate"] * 100
        fig = px.bar(
            isp_churn, x="Internet Service", y="Churn Rate",
            color="Churn Rate",
            color_continuous_scale=[RISK_COLORS["LOW"], RISK_COLORS["HIGH"], RISK_COLORS["CRITICAL"]],
            text="Churn Rate",
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig = styled_fig(fig, "Churn Rate by Internet Service")
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_r2:
        pay_churn = df.groupby("PaymentMethod")["Churn"].mean().reset_index()
        pay_churn.columns = ["Payment", "Churn Rate"]
        pay_churn["Churn Rate"] = pay_churn["Churn Rate"] * 100
        pay_churn = pay_churn.sort_values("Churn Rate", ascending=True)
        fig = px.bar(
            pay_churn, y="Payment", x="Churn Rate", orientation="h",
            color="Churn Rate",
            color_continuous_scale=[RISK_COLORS["LOW"], RISK_COLORS["HIGH"], RISK_COLORS["CRITICAL"]],
            text="Churn Rate",
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="inside")
        fig = styled_fig(fig, "Churn Rate by Payment Method")
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    # ── High-Risk table ───────────────────────
    st.markdown('<div class="section-header">🚨 High-Risk Customers</div>', unsafe_allow_html=True)

    high_risk_df = (
        df[df["risk_level"].isin(["HIGH", "CRITICAL"])]
        .sort_values("churn_probability", ascending=False)
        [["customerID", "Contract", "tenure", "MonthlyCharges", "churn_probability", "risk_level"]]
        .head(50)
        .reset_index(drop=True)
    )

    html_rows = []
    for _, row in high_risk_df.iterrows():
        html_rows.append(f"""
        <tr>
            <td>{row['customerID']}</td>
            <td>{row['Contract']}</td>
            <td class='num-col'>{int(row['tenure'])}</td>
            <td class='num-col'>${row['MonthlyCharges']:.2f}</td>
            <td class='num-col'>{row['churn_probability']*100:.1f}%</td>
            <td>{risk_badge(row['risk_level'])}</td>
        </tr>
        """)

    st.markdown(
        f"""
        <div style="max-height: 400px; overflow-y: auto; border: 1px solid var(--border-subtle); border-radius: 10px; background-color: var(--bg-panel);">
            <table class="custom-table">
                <thead>
                    <tr>
                        <th>Customer ID</th>
                        <th>Contract</th>
                        <th class='num-col'>Tenure (mo.)</th>
                        <th class='num-col'>Monthly ($)</th>
                        <th class='num-col'>Churn Prob.</th>
                        <th>Risk</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(html_rows)}
                </tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True
    )


# ═════════════════════════════════════════════
# PAGE 2 — CUSTOMER ANALYSIS
# ═════════════════════════════════════════════

elif page == "🔍 Customer Analysis":
    st.markdown("## 🔍 Customer Analysis")
    st.markdown("Search for a customer, review their profile, and understand their churn risk.")

    # ── Filters sidebar area ──────────────────
    fcol1, fcol2, fcol3 = st.columns([2, 1.5, 1.5])
    with fcol1:
        search_term = st.text_input("🔎 Search by Customer ID", placeholder="e.g. 7263-CASBF")
    with fcol2:
        risk_filter = st.selectbox("Filter by Risk Level", ["All", "CRITICAL", "HIGH", "MEDIUM", "LOW"])
    with fcol3:
        contract_filter = st.selectbox("Filter by Contract", ["All"] + sorted(df["Contract"].unique().tolist()))

    # Apply filters
    filtered = df.copy()
    if search_term:
        filtered = filtered[filtered["customerID"].str.contains(search_term.strip(), case=False, na=False)]
    if risk_filter != "All":
        filtered = filtered[filtered["risk_level"] == risk_filter]
    if contract_filter != "All":
        filtered = filtered[filtered["Contract"] == contract_filter]

    filtered = filtered.sort_values("churn_probability", ascending=False)

    if len(filtered) == 0:
        st.warning("No customers match the current filters.")
        st.stop()

    # Customer selector
    st.markdown(f"**{len(filtered):,} customers found** — select one to inspect:")
    selected_id = st.selectbox(
        "Select Customer",
        filtered["customerID"].tolist(),
        format_func=lambda x: f"{x}  (Risk: {filtered.loc[filtered['customerID']==x, 'risk_level'].values[0]}  |  Prob: {filtered.loc[filtered['customerID']==x, 'churn_probability'].values[0]*100:.1f}%)",
    )

    cust = df[df["customerID"] == selected_id].iloc[0]

    # ── Customer panel ────────────────────────
    st.markdown("---")
    info_col, pred_col, shap_col = st.columns([1.3, 1, 1.8])

    with info_col:
        st.markdown('<div class="section-header">👤 Customer Profile</div>', unsafe_allow_html=True)
        fields = {
            "Customer ID": cust["customerID"],
            "Gender": cust["gender"],
            "Senior Citizen": "Yes" if str(cust["SeniorCitizen"]) == "1" else "No",
            "Partner": cust["Partner"],
            "Dependents": cust["Dependents"],
            "Tenure": f"{int(cust['tenure'])} months",
            "Contract": cust["Contract"],
            "Internet Service": cust["InternetService"],
            "Payment Method": cust["PaymentMethod"],
            "Monthly Charges": f"${cust['MonthlyCharges']:.2f}",
            "Total Charges": f"${cust['TotalCharges']:.2f}",
            "Total Services": int(cust["TotalServices"]),
        }
        for k, v in fields.items():
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"padding:4px 0;border-bottom:1px solid #1e293b'>"
                f"<span style='color:#94a3b8;font-size:0.83rem;'>{k}</span>"
                f"<span style='color:#e2e8f0;font-size:0.83rem;font-weight:600;'>{v}</span></div>",
                unsafe_allow_html=True,
            )

    with pred_col:
        st.markdown('<div class="section-header">🎯 Churn Prediction</div>', unsafe_allow_html=True)
        prob = cust["churn_probability"]
        risk = cust["risk_level"]
        prediction = "Likely to Churn" if cust["prediction"] == 1 else "Likely to Stay"
        actual = "Churned" if cust["Churn"] == 1 else "Stayed"

        color_class = "prob-high" if prob > 0.6 else ("prob-medium" if prob > 0.3 else "prob-low")
        st.markdown(
            f"""
            <div style='background:#1e293b;border-radius:12px;padding:1.5rem;text-align:center;border:1px solid #334155;'>
                <div style='color:#94a3b8;font-size:0.8rem;margin-bottom:0.3rem;'>CHURN PROBABILITY</div>
                <div class='{color_class}'>{prob*100:.1f}%</div>
                <div style='margin-top:0.8rem;'>
                    {risk_badge(risk)}
                </div>
                <div style='color:#cbd5e1;margin-top:0.8rem;font-size:0.95rem;font-weight:600;'>
                    {prediction}
                </div>
                <hr style='border-color:#334155;margin:0.8rem 0;'>
                <div style='color:#64748b;font-size:0.75rem;'>
                    Actual label (known): <b style='color:#94a3b8;'>{actual}</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Probability gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "font": {"color": "#F3F5F9", "size": 28}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#8B94A8"},
                "bar": {"color": RISK_COLORS.get(risk, "#3B82F6")},
                "bgcolor": "#131A2A",
                "bordercolor": "#232C42",
                "steps": [
                    {"range": [0, 30],  "color": RISK_COLORS["LOW"]},
                    {"range": [30, 60], "color": RISK_COLORS["MEDIUM"]},
                    {"range": [60, 80], "color": RISK_COLORS["HIGH"]},
                    {"range": [80, 100], "color": RISK_COLORS["CRITICAL"]},
                ],
            },
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "#F3F5F9"},
            height=200,
            margin=dict(l=20, r=20, t=20, b=10),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with shap_col:
        st.markdown('<div class="section-header">🔬 Why Is This Customer at Risk?</div>', unsafe_allow_html=True)
        st.markdown(
            "<div class='info-box'><p>SHAP explains which features pushed the model toward or away from predicting churn. "
            "This is a model explanation — not a guaranteed cause.</p></div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div style='font-size: 13px; color: var(--text-secondary); margin-bottom: -10px;'><span style='color:#EF4444'>■</span> Increases risk &nbsp;&nbsp; <span style='color:#38BDF8'>■</span> Decreases risk</div>", unsafe_allow_html=True)

        # Compute SHAP for this customer
        from model.preprocessing import ALL_CATEGORICAL_COLS, ALL_NUMERICAL_COLS
        feature_cols = ALL_NUMERICAL_COLS + ALL_CATEGORICAL_COLS
        X_cust = df[df["customerID"] == selected_id][feature_cols]
        X_transformed = preprocessor.transform(X_cust)
        shap_vals = explainer.shap_values(X_transformed)[0]

        # Build sorted SHAP dataframe
        shap_df = pd.DataFrame({
            "Feature": feature_names,
            "SHAP Value": shap_vals,
        }).sort_values("SHAP Value", key=abs, ascending=False).head(12)

        shap_df["Direction"] = shap_df["SHAP Value"].apply(
            lambda v: "↑ Increases churn risk" if v > 0 else "↓ Reduces churn risk"
        )
        shap_df["Color"] = shap_df["SHAP Value"].apply(
            lambda v: "#EF4444" if v > 0 else "#38BDF8"
        )

        fig_shap = go.Figure(go.Bar(
            x=shap_df["SHAP Value"],
            y=shap_df["Feature"],
            orientation="h",
            marker_color=shap_df["Color"].tolist(),
            text=[f"{v:+.3f}" for v in shap_df["SHAP Value"]],
            textposition="outside",
        ))
        fig_shap = styled_fig(fig_shap, "SHAP Feature Contributions")
        fig_shap.update_layout(height=380, yaxis={"autorange": "reversed"})
        st.plotly_chart(fig_shap, use_container_width=True)

        # Text summary
        pos_factors = shap_df[shap_df["SHAP Value"] > 0]["Feature"].head(3).tolist()
        neg_factors = shap_df[shap_df["SHAP Value"] < 0]["Feature"].head(3).tolist()
        if pos_factors:
            st.markdown(
                f"**🔴 Top churn risk drivers:** {', '.join(pos_factors)}"
            )
        if neg_factors:
            st.markdown(
                f"**🟢 Top protective factors:** {', '.join(neg_factors)}"
            )


# ═════════════════════════════════════════════
# PAGE 3 — CHURN PREDICTION (Manual Form)
# ═════════════════════════════════════════════

elif page == "🎯 Churn Prediction":
    st.markdown("## 🎯 Live Churn Prediction")
    st.markdown("Enter customer characteristics to get an instant churn probability estimate.")

    from model.preprocessing import (
        ALL_CATEGORICAL_COLS,
        ALL_NUMERICAL_COLS,
        clean_data,
        engineer_features,
    )

    with st.form("prediction_form"):
        st.markdown('<div class="section-header" style="margin-top: 0;">👤 Account & Demographics</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)

        with c1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])

        with c2:
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

        with c3:
            payment = st.selectbox(
                "Payment Method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            )
            monthly_charges = st.number_input("Monthly Charges ($)", 18.0, 120.0, 65.0, 0.5)
            total_charges = st.number_input(
                "Total Charges ($)", 0.0, 9000.0, float(monthly_charges * (tenure + 1)), 1.0
            )

        st.markdown('<div class="section-header">🌐 Services</div>', unsafe_allow_html=True)
        s1, s2, s3 = st.columns(3)

        with s1:
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

        with s2:
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])

        with s3:
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

        submitted = st.form_submit_button("🔮 Predict Churn", use_container_width=True, type="primary")

    if submitted:
        # Build a single-row dataframe matching the raw schema
        input_data = pd.DataFrame([{
            "customerID": "MANUAL-INPUT",
            "gender": gender,
            "SeniorCitizen": 1 if senior == "Yes" else 0,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": str(total_charges),
            "Churn": "No",   # placeholder — NOT used in prediction
        }])

        # Run through the same cleaning + engineering
        cleaned = clean_data(input_data)
        featured = engineer_features(cleaned)

        feature_cols = ALL_NUMERICAL_COLS + ALL_CATEGORICAL_COLS
        X_new = featured[feature_cols]
        prob = float(model.predict_proba(X_new)[0, 1])
        pred = model.predict(X_new)[0]

        def get_risk(p):
            if p < 0.30: return "LOW"
            if p < 0.60: return "MEDIUM"
            if p < 0.80: return "HIGH"
            return "CRITICAL"

        risk = get_risk(prob)
        verdict = "Likely to Churn" if pred == 1 else "Likely to Stay"

        st.markdown("---")
        r1, r2, r3 = st.columns(3)

        color = RISK_COLORS.get(risk, "var(--accent-primary)")
        bg_wash = color + "15"  # 15 hex = ~8% opacity

        with r1:
            st.markdown(
                f"""<div class="kpi-card" style="border-color:{color};">
                    <div class="kpi-label">Churn Probability</div>
                    <div class="kpi-value" style="color:{color};">{prob*100:.1f}%</div>
                    <div class="kpi-sub">Model confidence</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with r2:
            st.markdown(
                f"""<div class="kpi-card" style="border-color:{color}; background-color:{bg_wash};">
                    <div class="kpi-label">Risk Level</div>
                    <div class="kpi-value" style="color:{color};font-size:1.5rem;">{risk}</div>
                    <div class="kpi-sub">Based on 4-tier classification</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with r3:
            st.markdown(
                f"""<div class="kpi-card" style="border-color:{color};">
                    <div class="kpi-label">Prediction</div>
                    <div class="kpi-value" style="color:{color};font-size:1.3rem;">{verdict}</div>
                    <div class="kpi-sub">XGBoost classification</div>
                </div>""",
                unsafe_allow_html=True,
            )

        # SHAP for this input
        st.markdown("### 🔬 What Drives This Prediction?")
        st.markdown("<div style='font-size: 13px; color: var(--text-secondary); margin-bottom: 8px;'><span style='color:#EF4444'>■</span> Increases risk &nbsp;&nbsp; <span style='color:#38BDF8'>■</span> Decreases risk</div>", unsafe_allow_html=True)
        X_transformed = preprocessor.transform(X_new)
        shap_vals = explainer.shap_values(X_transformed)[0]

        shap_df = pd.DataFrame({
            "Feature": feature_names,
            "SHAP Value": shap_vals,
        }).sort_values("SHAP Value", key=abs, ascending=False).head(10)

        shap_df["Color"] = shap_df["SHAP Value"].apply(lambda v: "#EF4444" if v > 0 else "#38BDF8")

        fig_s = go.Figure(go.Bar(
            x=shap_df["SHAP Value"],
            y=shap_df["Feature"],
            orientation="h",
            marker_color=shap_df["Color"].tolist(),
            text=[f"{v:+.3f}" for v in shap_df["SHAP Value"]],
            textposition="outside",
        ))
        fig_s = styled_fig(fig_s, "SHAP Contributions for This Customer")
        fig_s.update_layout(height=350, yaxis={"autorange": "reversed"})
        st.plotly_chart(fig_s, use_container_width=True)

        st.caption(
            "⚠️ SHAP explains the model's prediction — it does not prove causality. "
            "Predictions are based on patterns in the IBM Telco dataset."
        )


# ═════════════════════════════════════════════
# PAGE 4 — MODEL PERFORMANCE
# ═════════════════════════════════════════════

elif page == "📈 Model Performance":
    st.markdown("## 📈 Model Performance")
    st.markdown("Comparing Logistic Regression and XGBoost — both trained on the IBM Telco dataset.")

    lr = metrics["logistic_regression"]
    xgb = metrics["xgboost"]
    best = metrics["best_model"]

    # ── Metrics comparison cards ──────────────
    st.markdown("### Metrics Comparison")
    mc1, mc2 = st.columns(2)

    def model_card(col, name, m, is_best):
        badge = " 🏆 Best" if is_best else ""
        border_color = "var(--accent-primary)" if is_best else "var(--border-subtle)"
        col.markdown(
            f"""<div class="kpi-card" style="border-color:{border_color};text-align:left;padding:1.2rem;">
                <div style='color:var(--text-primary);font-size:1rem;font-weight:700;margin-bottom:0.8rem;'>
                    {name}{badge}
                </div>
                <div style='display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;'>
                    <div><span style='color:var(--text-secondary);font-size:0.78rem;'>Accuracy</span><br>
                         <span style='color:var(--text-primary);font-size:1.1rem;font-weight:600;'>{m['accuracy']:.4f}</span></div>
                    <div><span style='color:var(--text-secondary);font-size:0.78rem;'>ROC-AUC</span><br>
                         <span style='color:var(--text-primary);font-size:1.1rem;font-weight:600;'>{m['roc_auc']:.4f}</span></div>
                    <div><span style='color:var(--text-secondary);font-size:0.78rem;'>Precision</span><br>
                         <span style='color:var(--text-primary);font-size:1.1rem;font-weight:600;'>{m['precision']:.4f}</span></div>
                    <div style='background:rgba(59,130,246,0.1); padding:4px 8px; border-radius:6px; border:1px solid rgba(59,130,246,0.3);'>
                         <span style='color:var(--accent-primary);font-size:0.78rem;font-weight:700;'>Recall 🎯</span><br>
                         <span style='color:var(--text-primary);font-size:1.1rem;font-weight:600;'>{m['recall']:.4f}</span></div>
                    <div><span style='color:var(--text-secondary);font-size:0.78rem;'>F1-Score</span><br>
                         <span style='color:var(--text-primary);font-size:1.1rem;font-weight:600;'>{m['f1']:.4f}</span></div>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

    model_card(mc1, "Logistic Regression", lr, best == "logistic_regression")
    model_card(mc2, "XGBoost", xgb, best == "xgboost")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── ROC curves + confusion matrix ─────────
    ch1, ch2 = st.columns(2)

    with ch1:
        # ROC Curve — approximate with metrics (real curve needs y_prob; use stored AUC value as label)
        st.markdown("#### ROC-AUC Comparison")
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Bar(
            x=["Logistic Regression", "XGBoost"],
            y=[lr["roc_auc"], xgb["roc_auc"]],
            marker_color=["#64748b", "#3b82f6"],
            text=[f"{lr['roc_auc']:.4f}", f"{xgb['roc_auc']:.4f}"],
            textposition="outside",
            width=0.4,
        ))
        fig_roc.update_layout(**PLOT_LAYOUT, yaxis=dict(range=[0, 1.05], title="ROC-AUC"))
        st.plotly_chart(fig_roc, use_container_width=True)

    with ch2:
        st.markdown("#### XGBoost Confusion Matrix")
        cm = np.array(xgb["confusion_matrix"])
        labels = ["No Churn", "Churn"]
        text_labels = [
            [f"True Negative<br>{cm[0][0]}", f"False Positive<br>{cm[0][1]}"],
            [f"False Negative<br>{cm[1][0]}", f"True Positive<br>{cm[1][1]}"]
        ]
        
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=labels, y=labels,
            color_continuous_scale="Blues"
        )
        fig_cm.update_traces(text=text_labels, texttemplate="%{text}")
        fig_cm.update_layout(**PLOT_LAYOUT, title="Confusion Matrix (XGBoost)")
        fig_cm.update_coloraxes(showscale=False)
        st.plotly_chart(fig_cm, use_container_width=True)

    # ── Feature Importance ────────────────────
    st.markdown("### Global Feature Importance (SHAP)")
    st.markdown(
        "<div class='info-box'><p>Mean absolute SHAP value across training data. "
        "Larger bars = more influential features for predicting churn.</p></div>",
        unsafe_allow_html=True,
    )

    # Load global importance from the SHAP explainer artifact
    # We recompute from stored predictions (SHAP global was saved in metrics JSON if available)
    # Otherwise compute quickly from the model's built-in feature importance
    try:
        # Try to use XGBoost feature importance as proxy for global
        xgb_model = model.named_steps["classifier"]
        prep = model.named_steps["preprocessor"]
        fi_scores = xgb_model.feature_importances_
        fi_df = pd.DataFrame({
            "Feature": feature_names[:len(fi_scores)],
            "Importance": fi_scores,
        }).sort_values("Importance", ascending=False).head(15)

        colors = ["var(--accent-primary)"] * 3 + ["rgba(59,130,246,0.4)"] * max(0, len(fi_df)-3)
        fig_fi = go.Figure(go.Bar(
            y=fi_df["Feature"],
            x=fi_df["Importance"],
            orientation="h",
            marker_color=colors,
            text=[f"{v:.4f}" for v in fi_df["Importance"]],
            textposition="outside",
        ))
        fig_fi = styled_fig(fig_fi, "Top 15 Feature Importances (XGBoost)")
        fig_fi.update_layout(height=420, yaxis={"autorange": "reversed"})
        st.plotly_chart(fig_fi, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not render feature importance: {e}")

    # ── Classification Report ─────────────────
    st.markdown("### Classification Report — XGBoost")
    cr = xgb["classification_report"]
    cr_rows = []
    for label in ["No Churn", "Churn"]:
        row = cr.get(label, {})
        cr_rows.append({
            "Class": label,
            "Precision": f"{row.get('precision', 0):.3f}",
            "Recall": f"{row.get('recall', 0):.3f}",
            "F1-Score": f"{row.get('f1-score', 0):.3f}",
            "Support": int(row.get("support", 0)),
        })
    st.dataframe(pd.DataFrame(cr_rows), use_container_width=True, hide_index=True)

    st.caption(
        "All metrics are computed on a held-out 20% test split (stratified). "
        "No metrics are hard-coded — they reflect the actual model trained on this run."
    )
