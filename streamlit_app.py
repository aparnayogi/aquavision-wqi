"""
💧 AquaVision — Water Quality Intelligence
Streamlit version for cloud deployment
Run: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings, os, io
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import joblib

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# ── Page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="AquaVision | Water Quality AI",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ────────────────────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
*, *::before, *::after { box-sizing: border-box; }

/* Background */
.stApp {
    background: linear-gradient(135deg, #080c14 0%, #0f1825 100%) !important;
}
.main .block-container {
    padding: 2rem 2.5rem !important;
    max-width: 1400px !important;
}

/* Text */
h1,h2,h3,h4,h5,h6 {
    font-family: 'Space Mono', monospace !important;
    color: #dde3ef !important;
    font-weight: 700 !important;
}
p, div, span, label {
    font-family: 'DM Sans', sans-serif !important;
    color: #7d8a9e !important;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: #0e1420 !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 12px !important;
    padding: 20px !important;
    transition: all 0.2s !important;
}
[data-testid="metric-container"]:hover {
    border-color: rgba(0,196,255,0.3) !important;
    transform: translateY(-2px) !important;
}
[data-testid="metric-container"] [data-testid="stMetricLabel"] p {
    color: #7d8a9e !important; font-size: 12px !important;
    text-transform: uppercase !important; letter-spacing: 0.8px !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #00c4ff !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 2rem !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #0072ff, #00c4ff) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-weight: 600 !important;
    padding: 0.6rem 1.8rem !important;
    font-family: 'DM Sans', sans-serif !important;
    transition: all 0.2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; transform: translateY(-1px) !important; }

/* Download button */
.stDownloadButton > button {
    background: transparent !important;
    color: #00c4ff !important;
    border: 1px solid rgba(0,196,255,0.4) !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* Inputs */
.stNumberInput input, .stTextInput input {
    background: #151d2e !important; color: #dde3ef !important;
    border: 1px solid rgba(255,255,255,0.11) !important;
    border-radius: 8px !important;
    font-family: 'Space Mono', monospace !important;
}
.stNumberInput input:focus, .stTextInput input:focus {
    border-color: #00c4ff !important;
    box-shadow: 0 0 0 3px rgba(0,196,255,0.1) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #0e1420 !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 4px !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 8px !important;
    color: #7d8a9e !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    padding: 8px 16px !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(0,196,255,0.1) !important;
    color: #00c4ff !important;
}

/* Dataframe */
.stDataFrame { border-radius: 12px !important; border: 1px solid rgba(255,255,255,0.08) !important; }
[data-testid="stDataFrameResizable"] { background: #0e1420 !important; }

/* Alerts */
.stSuccess { background: rgba(31,216,160,0.1) !important; border-left: 4px solid #1fd8a0 !important; border-radius: 8px !important; }
.stError   { background: rgba(242,77,107,0.1)  !important; border-left: 4px solid #f24d6b !important; border-radius: 8px !important; }
.stWarning { background: rgba(245,166,35,0.1)  !important; border-left: 4px solid #f5a623 !important; border-radius: 8px !important; }
.stInfo    { background: rgba(0,196,255,0.08)   !important; border-left: 4px solid #00c4ff !important; border-radius: 8px !important; }

/* Sidebar hide */
[data-testid="collapsedControl"] { display: none !important; }

/* Divider */
hr { border-color: rgba(255,255,255,0.06) !important; margin: 1.5rem 0 !important; }
</style>
""", unsafe_allow_html=True)


# ── Constants ───────────────────────────────────────────────────────
FEATURES = ["ph", "dissolved_oxygen", "turbidity", "conductivity", "bod", "nitrates", "total_coliform"]
MODEL_PATH = "backend/model.pkl"


# ── Helpers ─────────────────────────────────────────────────────────
def wqi_label(score):
    if score >= 90:   return "Excellent", "#1fd8a0"
    elif score >= 70: return "Good",      "#00c4ff"
    elif score >= 50: return "Poor",      "#f5a623"
    elif score >= 25: return "Very Poor", "#f24d6b"
    return "Unsuitable", "#7a1228"


def get_recommendations(data: dict) -> dict:
    ph        = float(data.get("ph",               7.0))
    do        = float(data.get("dissolved_oxygen",  7.0))
    turbidity = float(data.get("turbidity",         1.0))
    bod       = float(data.get("bod",               2.0))
    cond      = float(data.get("conductivity",    300.0))
    nitrates  = float(data.get("nitrates",          5.0))
    coliform  = float(data.get("total_coliform",    0.0))

    items = []
    def add(param, val, unit, status, msg, action, icon):
        items.append({"parameter":param,"value":round(val,2),"unit":unit,"status":status,"message":msg,"action":action,"icon":icon})

    # pH
    if ph < 5.5:   add("pH",ph,"","danger",f"pH {ph} is critically acidic.","Add lime or soda ash.","⚗️")
    elif ph < 6.5: add("pH",ph,"","warn",  f"pH {ph} is mildly acidic.","Add alkaline buffer.","⚗️")
    elif ph > 9.0: add("pH",ph,"","danger",f"pH {ph} is strongly alkaline.","Inject CO₂ or mild acid.","⚗️")
    elif ph > 8.5: add("pH",ph,"","warn",  f"pH {ph} is mildly alkaline.","Consider mild acid treatment.","⚗️")
    else:          add("pH",ph,"","ok",    f"pH {ph} is within safe range.","No action needed.","⚗️")

    # DO
    if do < 2:   add("Dissolved O₂",do,"mg/L","danger","DO critically low — hypoxic!","Install emergency aeration.","💨")
    elif do < 5: add("Dissolved O₂",do,"mg/L","warn",  "DO is low — stress for aquatic life.","Increase aeration.","💨")
    else:        add("Dissolved O₂",do,"mg/L","ok",    "DO is healthy.","No action needed.","💨")

    # Turbidity
    if turbidity > 15:  add("Turbidity",turbidity,"NTU","danger","Very high turbidity — water is unsafe.","Apply coagulation + sand filtration.","🔵")
    elif turbidity > 4: add("Turbidity",turbidity,"NTU","warn",  "Turbidity exceeds WHO limit (4 NTU).","Use pre-filtration.","🔵")
    else:               add("Turbidity",turbidity,"NTU","ok",    "Turbidity is acceptable.","No action needed.","🔵")

    # BOD
    if bod > 6:   add("BOD",bod,"mg/L","danger","Heavy organic pollution.","Apply biological treatment.","🌿")
    elif bod > 3: add("BOD",bod,"mg/L","warn",  "Moderate organic load.","Monitor organic waste.","🌿")
    else:         add("BOD",bod,"mg/L","ok",    "BOD within safe limits.","No action needed.","🌿")

    # Conductivity
    if cond > 1000:  add("Conductivity",cond,"µS/cm","danger","Excessively high dissolved salts.","Use Reverse Osmosis.","🧂")
    elif cond > 600: add("Conductivity",cond,"µS/cm","warn",  "Elevated conductivity.","Check for salt intrusion.","🧂")
    else:            add("Conductivity",cond,"µS/cm","ok",    "Conductivity is normal.","No action needed.","🧂")

    # Nitrates
    if nitrates > 10:  add("Nitrates",nitrates,"mg/L","danger","Exceeds WHO limit (10 mg/L).","Use ion exchange or RO.","⚠️")
    elif nitrates > 5: add("Nitrates",nitrates,"mg/L","warn",  "Elevated — possible agricultural runoff.","Increase monitoring.","⚠️")
    else:              add("Nitrates",nitrates,"mg/L","ok",    "Nitrates within safe limits.","No action needed.","⚠️")

    # Coliform
    if coliform > 2:   add("Total Coliform",coliform,"CFU","danger","Serious microbial contamination!","Shock chlorination + UV treatment.","🦠")
    elif coliform > 0: add("Total Coliform",coliform,"CFU","warn",  "Coliform detected.","Apply chlorination.","🦠")
    else:              add("Total Coliform",coliform,"CFU","ok",    "No coliform detected.","Maintain disinfection.","🦠")

    danger = sum(1 for i in items if i["status"]=="danger")
    warn   = sum(1 for i in items if i["status"]=="warn")
    ok     = sum(1 for i in items if i["status"]=="ok")

    if danger >= 2:   summary, score = "Danger", max(0,  30 - danger*8)
    elif danger == 1: summary, score = "Danger", max(20, 45 - warn*5)
    elif warn >= 2:   summary, score = "Caution",max(40, 65 - warn*5)
    elif warn == 1:   summary, score = "Caution",72
    else:             summary, score = "Safe",   min(100,85+ok*2)

    return {"summary":summary,"score":score,"counts":{"danger":danger,"warn":warn,"ok":ok},"items":items}


def fallback_wqi(d):
    ph_s   = max(0, 100 - abs(d.get("ph",7)-7)*20)
    do_s   = min(100, d.get("dissolved_oxygen",7)*12)
    turb_s = max(0, 100 - d.get("turbidity",1)*5)
    bod_s  = max(0, 100 - d.get("bod",2)*10)
    col_s  = 100 if d.get("total_coliform",0)==0 else 0
    return ph_s*0.2 + do_s*0.25 + turb_s*0.2 + bod_s*0.2 + col_s*0.15


# ── Dataset generation ───────────────────────────────────────────────
@st.cache_data
def generate_dataset(n=2000):
    np.random.seed(42)
    df = pd.DataFrame({
        "ph":               np.random.uniform(5.0, 9.5,  n),
        "dissolved_oxygen": np.random.uniform(1.0, 14.0, n),
        "turbidity":        np.random.uniform(0.1, 25.0, n),
        "conductivity":     np.random.uniform(50,  1500, n),
        "bod":              np.random.uniform(0.5, 12.0, n),
        "nitrates":         np.random.uniform(0.1, 20.0, n),
        "total_coliform":   np.random.uniform(0,   5.0,  n),
        "latitude":         np.random.uniform(8.0, 35.0, n),
        "longitude":        np.random.uniform(68., 97.0, n),
    })
    ph_s   = np.clip(100 - np.abs(df.ph - 7.5)*25,       0,100)
    do_s   = np.clip(df.dissolved_oxygen*11,               0,100)
    turb_s = np.clip(100 - df.turbidity*4.5,              0,100)
    cond_s = np.clip(100 - (df.conductivity-50)*0.08,     0,100)
    bod_s  = np.clip(100 - df.bod*12,                      0,100)
    nit_s  = np.clip(100 - df.nitrates*6,                  0,100)
    col_s  = np.clip(100 - df.total_coliform*20,           0,100)
    df["wqi"] = (ph_s*.20 + do_s*.22 + turb_s*.18 + cond_s*.10 + bod_s*.15 + nit_s*.08 + col_s*.07)
    df["wqi"] += np.random.normal(0, 1.5, n)
    df["wqi"]  = df["wqi"].clip(0,100).round(2)
    return df.round(3)


# ── Model training ───────────────────────────────────────────────────
@st.cache_resource
def train_models(df):
    X = df[FEATURES]
    y = df["wqi"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest":     RandomForestRegressor(n_estimators=100, random_state=42),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, verbosity=0)

    results = {}
    best_model, best_r2, best_name = None, -999, ""
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2   = round(r2_score(y_test, y_pred), 4)
        rmse = round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4)
        results[name] = {"r2": r2, "rmse": rmse}
        if r2 > best_r2:
            best_r2, best_model, best_name = r2, model, name

    return best_model, best_name, results


def predict_single(model, input_dict):
    values = [float(input_dict.get(f, 0)) for f in FEATURES]
    X = np.array(values).reshape(1,-1)
    wqi = float(model.predict(X)[0])
    wqi = max(0, min(100, wqi))
    imp = {}
    if hasattr(model, "feature_importances_"):
        for n,v in zip(FEATURES, model.feature_importances_):
            imp[n] = round(float(v), 4)
    else:
        coefs = np.abs(model.coef_)
        coefs = coefs / coefs.sum()
        for n,v in zip(FEATURES, coefs):
            imp[n] = round(float(v), 4)
    return round(wqi,2), imp


# ── Load data & train ─────────────────────────────────────────────
with st.spinner("⏳ Loading AquaVision… training models…"):
    df      = generate_dataset()
    model, model_name, eval_results = train_models(df)

# ── Hero banner ──────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#0072ff,#00c4ff);
            padding:2.5rem 2rem;border-radius:16px;text-align:center;
            margin-bottom:2rem;box-shadow:0 16px 40px rgba(0,114,255,0.25)">
  <div style="font-family:'Space Mono',monospace;font-size:2.4rem;
              font-weight:700;color:white;letter-spacing:-1px">💧 AquaVision</div>
  <div style="color:rgba(255,255,255,0.9);margin-top:0.5rem;
              font-size:1rem;font-family:'DM Sans',sans-serif">
    AI-Powered Water Quality Monitoring & Decision Support System
  </div>
</div>
""", unsafe_allow_html=True)

# ── Navigation tabs ───────────────────────────────────────────────────
tab_overview, tab_predict, tab_recos, tab_bulk, tab_model = st.tabs([
    "📊 Overview", "🤖 Predict WQI", "💡 Recommendations", "📤 Bulk Scanner", "🧠 Model Insight"
])


# ══════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════
with tab_overview:
    st.subheader("Dataset Overview")

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Records",    f"{len(df):,}",      "observations")
    c2.metric("Avg WQI Score",    f"{df.wqi.mean():.1f}", "quality index")
    c3.metric("Good Quality (≥70)", f"{(df.wqi>=70).sum():,}", "records")
    c4.metric("Needs Attention (<50)", f"{(df.wqi<50).sum():,}", "records")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Category Distribution")
        bins  = ["Excellent\n(≥90)","Good\n(70-89)","Poor\n(50-69)","Very Poor\n(25-49)","Unsuitable\n(<25)"]
        counts = [
            int((df.wqi>=90).sum()), int(((df.wqi>=70)&(df.wqi<90)).sum()),
            int(((df.wqi>=50)&(df.wqi<70)).sum()), int(((df.wqi>=25)&(df.wqi<50)).sum()),
            int((df.wqi<25).sum())
        ]
        fig = px.pie(
            names=bins, values=counts,
            color_discrete_sequence=["#1fd8a0","#00c4ff","#f5a623","#f24d6b","#7a1228"],
            hole=0.5,
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#7d8a9e", family="DM Sans"), height=320,
            legend=dict(font=dict(color="#7d8a9e")),
            margin=dict(t=20,b=20,l=20,r=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### Parameter Statistics")
        stats = []
        for f in FEATURES:
            stats.append({
                "Parameter": f.replace("_"," ").title(),
                "Mean":   round(df[f].mean(), 2),
                "Std":    round(df[f].std(),  2),
                "Min":    round(df[f].min(),  2),
                "Max":    round(df[f].max(),  2),
            })
        st.dataframe(pd.DataFrame(stats), use_container_width=True, hide_index=True, height=320)

    st.divider()
    st.markdown("#### WQI Score Distribution")
    fig2 = px.histogram(
        df, x="wqi", nbins=30,
        color_discrete_sequence=["#00c4ff"],
        labels={"wqi":"WQI Score","count":"Records"},
    )
    fig2.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#7d8a9e", family="DM Sans"), height=260,
        bargap=0.05, margin=dict(t=10,b=20,l=10,r=10),
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
    )
    st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 2 — PREDICT WQI
# ══════════════════════════════════════════════════════════════════════
with tab_predict:
    st.subheader("Predict Water Quality Index")

    col_form, col_result = st.columns([1,1])

    with col_form:
        st.markdown("**Enter Sensor Parameters**")
        ph   = st.number_input("pH Level",             0.0, 14.0,  7.2, 0.1, key="ph")
        do   = st.number_input("Dissolved Oxygen (mg/L)", 0.0, 20.0, 8.5, 0.1, key="do")
        turb = st.number_input("Turbidity (NTU)",      0.0,100.0,  2.1, 0.1, key="turb")
        cond = st.number_input("Conductivity (µS/cm)", 0.0,2000.0,320.0,1.0, key="cond")
        bod  = st.number_input("BOD (mg/L)",           0.0, 20.0,  1.8, 0.1, key="bod")
        nit  = st.number_input("Nitrates (mg/L)",      0.0, 50.0,  3.2, 0.1, key="nit")
        col  = st.number_input("Total Coliform (CFU)", 0.0, 10.0,  0.0, 0.1, key="col")
        predict_btn = st.button("🚀 Run Prediction", use_container_width=True)

    with col_result:
        if predict_btn:
            payload = {"ph":ph,"dissolved_oxygen":do,"turbidity":turb,
                       "conductivity":cond,"bod":bod,"nitrates":nit,"total_coliform":col}
            wqi_score, importance = predict_single(model, payload)
            label, color = wqi_label(wqi_score)

            # WQI Gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=wqi_score,
                domain={"x":[0,1],"y":[0,1]},
                title={"text":"WQI Score","font":{"color":"#7d8a9e","family":"DM Sans"}},
                number={"font":{"color":color,"family":"Space Mono","size":48}},
                gauge={
                    "axis":{"range":[0,100],"tickcolor":"#7d8a9e","tickfont":{"color":"#7d8a9e"}},
                    "bar":{"color":color,"thickness":0.25},
                    "bgcolor":"rgba(0,0,0,0)",
                    "borderwidth":0,
                    "steps":[
                        {"range":[0,25],  "color":"rgba(122,18,40,0.3)"},
                        {"range":[25,50], "color":"rgba(242,77,107,0.2)"},
                        {"range":[50,70], "color":"rgba(245,166,35,0.2)"},
                        {"range":[70,90], "color":"rgba(0,196,255,0.2)"},
                        {"range":[90,100],"color":"rgba(31,216,160,0.2)"},
                    ],
                    "threshold":{"line":{"color":color,"width":3},"thickness":0.75,"value":wqi_score}
                }
            ))
            fig_gauge.update_layout(
                height=280, margin=dict(t=40,b=10,l=20,r=20),
                paper_bgcolor="rgba(0,0,0,0)", font=dict(family="DM Sans")
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            st.markdown(f"""
            <div style="text-align:center;padding:10px;
                        background:rgba(0,0,0,0.2);border-radius:10px;
                        border:1px solid {color}44;margin-bottom:16px">
              <span style="color:{color};font-size:18px;font-weight:700;
                           font-family:'Space Mono',monospace">{label}</span>
              <span style="color:#7d8a9e;font-size:13px;margin-left:10px">
                Model: {model_name}
              </span>
            </div>""", unsafe_allow_html=True)

            # Feature importance bar
            st.markdown("**Feature Importance**")
            imp_df = pd.DataFrame(list(importance.items()), columns=["Feature","Importance"])
            imp_df = imp_df.sort_values("Importance", ascending=True)
            fig_imp = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                             color_discrete_sequence=["#00c4ff"])
            fig_imp.update_layout(
                height=240, margin=dict(t=10,b=10,l=10,r=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#7d8a9e", family="DM Sans"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
                yaxis=dict(gridcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig_imp, use_container_width=True)

            # store in session for recommendations tab
            st.session_state["last_payload"] = payload
            st.session_state["last_wqi"]     = wqi_score
        else:
            st.info("👈 Enter sensor values and click **Run Prediction**")


# ══════════════════════════════════════════════════════════════════════
# TAB 3 — RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════
with tab_recos:
    st.subheader("Smart Recommendations")

    # Use last predict payload if available, else show defaults
    default_payload = st.session_state.get("last_payload", {
        "ph":7.2,"dissolved_oxygen":8.5,"turbidity":2.1,
        "conductivity":320,"bod":1.8,"nitrates":3.2,"total_coliform":0
    })

    recos = get_recommendations(default_payload)
    summary  = recos["summary"]
    score    = recos["score"]
    counts   = recos["counts"]
    items    = recos["items"]

    sum_color = "#1fd8a0" if summary=="Safe" else "#f5a623" if summary=="Caution" else "#f24d6b"

    # Summary bar
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Overall Status", summary)
    c2.metric("Safety Score",   f"{score}/100")
    c3.metric("🔴 Critical",    counts["danger"])
    c4.metric("🟡 Caution",     counts["warn"])

    st.divider()

    # Cards in 2 columns
    col_a, col_b = st.columns(2)
    for i, item in enumerate(items):
        target = col_a if i % 2 == 0 else col_b
        color  = "#1fd8a0" if item["status"]=="ok" else "#f5a623" if item["status"]=="warn" else "#f24d6b"
        badge  = "✅ SAFE" if item["status"]=="ok" else "⚠️ CAUTION" if item["status"]=="warn" else "🚨 DANGER"
        with target:
            st.markdown(f"""
            <div style="background:#0e1420;border:1px solid {color}44;
                        border-left:3px solid {color};border-radius:12px;
                        padding:16px;margin-bottom:12px">
              <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
                <span style="font-size:22px">{item["icon"]}</span>
                <strong style="color:#dde3ef;font-size:14px">{item["parameter"]}</strong>
                <span style="color:{color};font-family:'Space Mono',monospace;
                             background:{color}18;padding:2px 8px;
                             border-radius:8px;font-size:11px">
                  {item["value"]} {item["unit"]}
                </span>
                <span style="color:{color};font-size:11px;font-weight:700;margin-left:auto">{badge}</span>
              </div>
              <div style="color:#7d8a9e;font-size:13px;margin-bottom:8px">{item["message"]}</div>
              <div style="background:{color}12;color:{color};font-size:12px;
                          padding:8px 12px;border-radius:7px">
                <strong>Action:</strong> {item["action"]}
              </div>
            </div>""", unsafe_allow_html=True)

    if "last_payload" not in st.session_state:
        st.info("💡 Go to **Predict WQI** tab and run a prediction to get recommendations for your specific values. Showing defaults above.")


# ══════════════════════════════════════════════════════════════════════
# TAB 4 — BULK SCANNER
# ══════════════════════════════════════════════════════════════════════
with tab_bulk:
    st.subheader("Bulk Water Quality Scanner")

    # Template download
    sample = pd.DataFrame({
        "location":["Site A","Site B","Site C"],
        "state":["Gujarat","Maharashtra","Karnataka"],
        "ph":[7.1,6.8,7.5],
        "dissolved_oxygen":[7.2,5.8,8.1],
        "turbidity":[2.5,8.2,1.8],
        "conductivity":[280,450,320],
        "bod":[1.5,3.2,2.1],
        "nitrates":[12.5,18.3,9.7],
        "total_coliform":[0,1,0],
        "latitude":[23.02,19.07,12.97],
        "longitude":[72.57,72.87,77.59],
    })
    col_dl, col_up, col_res = st.columns(3)
    with col_dl:
        st.download_button(
            "⬇ Download Template", sample.to_csv(index=False).encode(),
            "aquavision_template.csv", "text/csv", use_container_width=True
        )

    with col_up:
        uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

    st.divider()

    if uploaded:
        try:
            df_up = pd.read_csv(uploaded)
            required = ["ph","dissolved_oxygen","turbidity","conductivity","bod","nitrates","total_coliform"]
            missing  = [c for c in required if c not in df_up.columns]

            if missing:
                st.error(f"❌ Missing columns: {', '.join(missing)}")
            else:
                st.success(f"✅ File loaded: {len(df_up)} records, {len(df_up.columns)} columns")
                st.dataframe(df_up.head(5), use_container_width=True)

                if st.button("🚀 Scan & Predict All", use_container_width=True):
                    results = []
                    progress = st.progress(0)
                    for i, row in df_up.iterrows():
                        try:
                            inp = {f: float(row.get(f,0)) for f in required}
                            wqi_val, _ = predict_single(model, inp)
                            lab, _     = wqi_label(wqi_val)
                            results.append({
                                "Row":       i+1,
                                "Location":  row.get("location", f"Row {i+1}"),
                                "State":     row.get("state",    "—"),
                                "pH":        round(row.get("ph",0),2),
                                "DO":        round(row.get("dissolved_oxygen",0),1),
                                "Turbidity": round(row.get("turbidity",0),1),
                                "WQI Score": wqi_val,
                                "Category":  lab,
                            })
                        except:
                            results.append({"Row":i+1,"Location":f"Row {i+1}","WQI Score":"Error","Category":"—"})
                        progress.progress((i+1)/len(df_up))

                    res_df = pd.DataFrame(results)
                    scores_num = pd.to_numeric(res_df["WQI Score"], errors="coerce").dropna()

                    # Summary stats
                    sc1,sc2,sc3,sc4,sc5 = st.columns(5)
                    sc1.metric("Total Records",    len(res_df))
                    sc2.metric("Avg WQI",          f"{scores_num.mean():.1f}")
                    sc3.metric("Good (≥70)",        int((scores_num>=70).sum()))
                    sc4.metric("Needs Attention",   int((scores_num<50).sum()))
                    sc5.metric("Success Rate",      f"{int(len(scores_num)/len(res_df)*100)}%")

                    st.divider()
                    st.dataframe(res_df, use_container_width=True, height=400)

                    # Download results
                    with col_res:
                        st.download_button(
                            "⬇ Download Results", res_df.to_csv(index=False).encode(),
                            "aquavision_results.csv", "text/csv", use_container_width=True
                        )
        except Exception as e:
            st.error(f"❌ Error: {e}")
    else:
        st.info("⬆ Upload a CSV file above, or download the template first to see the required format.")


# ══════════════════════════════════════════════════════════════════════
# TAB 5 — MODEL INSIGHT
# ══════════════════════════════════════════════════════════════════════
with tab_model:
    st.subheader("Model Training & Evaluation")

    # Model comparison table
    st.markdown("#### Model Comparison")
    eval_df = pd.DataFrame([
        {"Model": k, "R² Score": v["r2"], "RMSE": v["rmse"],
         "Status": "✅ Best" if k == model_name else ""}
        for k, v in eval_results.items()
    ])
    st.dataframe(eval_df, use_container_width=True, hide_index=True)

    st.divider()

    # Feature importance chart
    st.markdown(f"#### Feature Importance — {model_name}")
    if hasattr(model, "feature_importances_"):
        imp_vals = model.feature_importances_
    else:
        imp_vals = np.abs(model.coef_) / np.abs(model.coef_).sum()

    imp_df = pd.DataFrame({"Feature": FEATURES, "Importance": imp_vals}).sort_values("Importance", ascending=True)
    fig_fi = px.bar(
        imp_df, x="Importance", y="Feature", orientation="h",
        color="Importance", color_continuous_scale=["#0072ff","#00c4ff"],
        labels={"Importance":"Importance Score"},
    )
    fig_fi.update_layout(
        height=340, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#7d8a9e", family="DM Sans"),
        coloraxis_showscale=False, margin=dict(t=10,b=10,l=10,r=10),
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig_fi, use_container_width=True)

    st.divider()

    # Sample predictions vs actual
    st.markdown("#### Predicted vs Actual (test sample)")
    sample_df = df.sample(200, random_state=1)
    X_samp = sample_df[FEATURES].values
    y_true = sample_df["wqi"].values
    y_pred = model.predict(X_samp)
    y_pred = np.clip(y_pred, 0, 100)

    fig_pva = px.scatter(
        x=y_true, y=y_pred,
        labels={"x":"Actual WQI","y":"Predicted WQI"},
        color_discrete_sequence=["#00c4ff"],
        opacity=0.6,
    )
    fig_pva.add_shape(type="line", x0=0,y0=0,x1=100,y1=100, line=dict(color="#f5a623",dash="dash",width=1))
    fig_pva.update_layout(
        height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#7d8a9e", family="DM Sans"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.04)"),
        margin=dict(t=10,b=10,l=10,r=10)
    )
    st.plotly_chart(fig_pva, use_container_width=True)


# ── Footer ──────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center;padding:1rem;color:#3d4a5e;font-size:13px;
            font-family:'DM Sans',sans-serif">
  💧 <strong style="color:#7d8a9e">AquaVision</strong> —
  AI Water Quality Monitoring System · Final Year Major Project
</div>
""", unsafe_allow_html=True)