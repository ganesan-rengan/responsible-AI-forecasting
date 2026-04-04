# -*- coding: utf-8 -*-
"""
app/app.py
-----------
Streamlit dashboard for the Responsible AI project.
Select a country to view its synthetic time series and LSTM forecast.
Run with: streamlit run app/app.py

Fixes applied:
  - Replaced deprecated use_container_width with width='stretch'
  - LSTM training wrapped in @st.cache_resource to avoid retraining on every country change
  - Dark theme applied to all matplotlib charts
  - Fairness bar chart x-axis label rotation fix
  - Sidebar navigation added
  - tight_layout applied to all figures
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ── Dark theme for all matplotlib charts ───────────────────────────
plt.style.use('dark_background')

# ── Config ─────────────────────────────────────────────────────────
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
DATA_PATH        = os.path.join(BASE_DIR, '..', 'data', 'AI_index_db.csv')
RESULTS_FAIRNESS = os.path.join(BASE_DIR, '..', 'results', 'fairness_summary.csv')

LOOKBACK = 4
EPOCHS   = 30
np.random.seed(42)

# ── Page config ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Responsible AI — AI Index Forecaster",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Sidebar navigation ─────────────────────────────────────────────
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🗂️ Dataset Preview", "🔮 Country Forecast", "⚖️ Fairness Audit"]
)

st.sidebar.markdown("---")
st.sidebar.caption("Data: AI Readiness Index\nModel: LSTM\nFairness: Aequitas")

# ── Main title ─────────────────────────────────────────────────────
st.title("Responsible AI — AI Index Forecaster")
st.markdown("**E-Governance AI Readiness Forecasting | Bias Detection across Region & Income Group**")
st.markdown("---")

# ── 1. Load data ───────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

# ── 2. Generate synthetic time series ──────────────────────────────
@st.cache_data
def build_all_ts(df):
    """Generate synthetic quarterly time series per country."""

    def generate_synthetic_ts(country_name, score_2024):
        dates  = pd.date_range(start='2015-01-01', periods=40, freq='Q')
        n      = len(dates)
        delta  = np.random.uniform(5, 15)
        start  = max(0, score_2024 - delta)
        trend    = np.linspace(start, score_2024, n)
        seasonal = 0.03 * score_2024 * np.sin(2 * np.pi * np.arange(n) / 4)
        noise    = np.random.normal(0, 0.01 * score_2024, n)
        y = np.maximum(trend + seasonal + noise, 0)
        return pd.DataFrame({'ds': dates, 'y': y, 'country': country_name})

    return pd.concat([
        generate_synthetic_ts(row['Country'], row['Total score'])
        for _, row in df.iterrows()
    ], ignore_index=True)

all_ts = build_all_ts(df)

# ── Helper: build sequences ────────────────────────────────────────
def make_seq(data, lb):
    X, y = [], []
    for i in range(lb, len(data)):
        X.append(data[i - lb:i])
        y.append(data[i])
    return np.array(X), np.array(y)

# ── FIX: LSTM cached so it doesn't retrain on every country change ─
@st.cache_resource
def train_lstm(X_tr_bytes, y_tr_bytes, shape):
    """
    Cache the trained LSTM model per unique training data.
    Accepts numpy arrays serialised to bytes for hashability.
    """
    X_tr = np.frombuffer(X_tr_bytes, dtype=np.float32).reshape(shape)
    y_tr = np.frombuffer(y_tr_bytes, dtype=np.float32).reshape(-1, 1)

    model = Sequential([
        LSTM(50, activation='relu', input_shape=(LOOKBACK, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_tr, y_tr, epochs=EPOCHS, verbose=0, validation_split=0.1)
    return model

# ══════════════════════════════════════════════════════════════════
# PAGE 1 — Dataset Preview
# ══════════════════════════════════════════════════════════════════
if page == "🗂️ Dataset Preview":
    st.subheader("🗂️ Raw Dataset Preview")
    st.markdown(f"**{len(df)} countries · {df.shape[1]} features**")

    # FIX: replaced use_container_width=True → width='stretch'
    st.dataframe(df, width='stretch')

    st.markdown("#### 📌 Feature Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Countries", len(df))
    col2.metric("Features", df.shape[1])
    col3.metric("Target Variable", "Total Score")

    st.markdown("#### 📊 Score Distribution by Region")
    if 'Region' in df.columns and 'Total score' in df.columns:
        fig0, ax0 = plt.subplots(figsize=(9, 4))
        regions = df['Region'].unique()
        colors  = ['#00BFFF', '#FF6B6B', '#FFD700', '#7CFC00', '#DA70D6']
        for i, region in enumerate(regions):
            subset = df[df['Region'] == region]['Total score']
            ax0.scatter([region] * len(subset), subset,
                        alpha=0.7, s=60, color=colors[i % len(colors)], label=region)
        ax0.set_xlabel("Region")
        ax0.set_ylabel("Total Score")
        ax0.set_title("AI Readiness Score Distribution by Region")
        ax0.tick_params(axis='x', rotation=20)
        fig0.tight_layout()
        st.pyplot(fig0)
        plt.close(fig0)

# ══════════════════════════════════════════════════════════════════
# PAGE 2 — Country Forecast
# ══════════════════════════════════════════════════════════════════
elif page == "🔮 Country Forecast":
    st.subheader("🔮 Country LSTM Forecast")

    country = st.selectbox("Select a country", sorted(df['Country'].unique()))

    country_data = (
        all_ts[all_ts['country'] == country]
        .sort_values('ds')
        .reset_index(drop=True)
    )

    values = country_data['y'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(values)

    split   = int(len(scaled) * 0.8)
    train_s = scaled[:split]
    test_s  = scaled[split:]

    X_tr, y_tr = make_seq(train_s, LOOKBACK)
    X_te, y_te = make_seq(test_s,  LOOKBACK)

    if len(X_tr) > 0 and len(X_te) > 0:

        # Serialise to bytes for cache key
        X_tr_f = X_tr.astype(np.float32)
        y_tr_f = y_tr.astype(np.float32)

        with st.spinner("Loading / training LSTM model..."):
            model = train_lstm(
                X_tr_f.tobytes(),
                y_tr_f.tobytes(),
                X_tr_f.shape
            )

        pred        = scaler.inverse_transform(model.predict(X_te, verbose=0))
        actual_test = scaler.inverse_transform(y_te.reshape(-1, 1))
        rmse        = np.sqrt(mean_squared_error(actual_test, pred))

        # Plot
        fig, ax = plt.subplots(figsize=(11, 4))
        ax.plot(country_data['ds'].values, values,
                label='Full Series', color='steelblue', linewidth=1.5)

        test_dates = country_data['ds'].values[split + LOOKBACK:]
        ax.plot(test_dates, actual_test,
                label='Actual (test)', color='orange', linewidth=2)
        ax.plot(test_dates, pred,
                label='LSTM Forecast', color='red', linestyle='--', linewidth=2)

        ax.set_title(f'{country} — LSTM Forecast', fontsize=13)
        ax.set_xlabel("Quarter")
        ax.set_ylabel("AI Readiness Score")
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Metrics row
        c1, c2, c3 = st.columns(3)
        c1.metric("Test RMSE", f"{rmse:.4f}")
        c2.metric("Train Quarters", split)
        c3.metric("Test Quarters", len(actual_test))

        st.info(
            "ℹ️ **Note:** Time series are synthetically generated from each country's "
            "2024 Total Score with realistic trend, seasonal, and noise components. "
            "Small test set (4–8 points) is a known limitation of single-snapshot data."
        )

    else:
        st.warning("Not enough data to train LSTM for this country.")

# ══════════════════════════════════════════════════════════════════
# PAGE 3 — Fairness Audit
# ══════════════════════════════════════════════════════════════════
elif page == "⚖️ Fairness Audit":
    st.subheader("⚖️ Responsible AI — AEQUITAS Fairness Evaluation")

    st.markdown("""
    The fairness audit evaluates whether the AI forecasting model treats all
    **regions** and **income groups** equally using three metrics:

    | Metric | Meaning |
    |--------|---------|
    | **PPR** | Predicted Positive Rate — how often the model predicts *high readiness* |
    | **FPR** | False Positive Rate — how often the model *wrongly* predicts high readiness |
    | **FNR** | False Negative Rate — how often the model *misses* a genuinely high-readiness country |
    """)

    try:
        fairness_df = pd.read_csv(RESULTS_FAIRNESS)

        st.markdown("#### 📋 Fairness Metrics Table")
        # FIX: replaced use_container_width=True → width='stretch'
        st.dataframe(fairness_df, width='stretch')

        if {'attribute_value', 'ppr', 'fpr', 'fnr'}.issubset(fairness_df.columns):

            # FIX: rotation + tight_layout for cramped labels
            fig2, ax2 = plt.subplots(figsize=(10, 5))

            plot_df = fairness_df.set_index("attribute_value")[["ppr", "fpr", "fnr"]]
            plot_df.plot(
                kind="bar",
                ax=ax2,
                color=['#00BFFF', '#FF6B6B', '#FFD700'],
                edgecolor='white',
                width=0.7
            )

            ax2.set_title("Fairness Metrics by Group (Region & Income)", fontsize=13)
            ax2.set_xlabel("Group")
            ax2.set_ylabel("Metric Value")
            ax2.tick_params(axis='x', rotation=45)   # FIX: was overlapping
            ax2.legend(title="Metric")
            ax2.axhline(0.5, color='white', linestyle=':', linewidth=0.8, alpha=0.4)
            fig2.tight_layout()                        # FIX: prevents label clipping
            st.pyplot(fig2)
            plt.close(fig2)


            # -- Verdict logic: reads from CSV verdict column if available,
            # falls back to threshold calculation if not present
            FPR_THRESHOLD = 0.20
            has_verdict_col = "verdict" in fairness_df.columns

            st.markdown("#### Fairness Verdict per Group")
            st.caption("Rule: FPR > 0.20 OR PPR = 0.000 = BIASED | Otherwise = FAIR")

            # Build verdict rows
            verdict_rows = []
            for _, row in fairness_df.iterrows():
                group   = str(row["attribute_value"])
                attr    = str(row["attribute_name"])
                ppr_val = float(row["ppr"]) if pd.notna(row["ppr"]) else 0.0
                fpr_val = float(row["fpr"]) if pd.notna(row["fpr"]) else 0.0

                # Use verdict from CSV if available, else calculate
                if has_verdict_col and pd.notna(row.get("verdict")):
                    is_biased = str(row["verdict"]).strip().upper() == "BIASED"
                else:
                    is_biased = (fpr_val > FPR_THRESHOLD) or (ppr_val == 0.0)

                if is_biased:
                    verdict = "BIASED"
                    reason  = f"FPR={fpr_val:.3f} over-predicted" if fpr_val > FPR_THRESHOLD else "PPR=0.000 never predicted positive"
                    bg      = "#4a1010"
                    border  = "#ef5350"
                    icon    = "BIASED"
                else:
                    verdict = "FAIR"
                    reason  = f"FPR={fpr_val:.3f} | PPR={ppr_val:.3f}"
                    bg      = "#0d3320"
                    border  = "#66bb6a"
                    icon    = "FAIR"

                verdict_rows.append({
                    "group": group, "attr": attr,
                    "ppr": ppr_val, "fpr": fpr_val,
                    "verdict": verdict, "reason": reason,
                    "bg": bg, "border": border, "icon": icon
                })


            # Render verdict cards — 4 per row
            chunk_size = 4
            for i in range(0, len(verdict_rows), chunk_size):
                chunk = verdict_rows[i:i+chunk_size]
                cols  = st.columns(len(chunk))
                for col, vr in zip(cols, chunk):
                    with col:
                        emoji  = "WARNING" if vr["verdict"] == "BIASED" else "OK"
                        label  = "BIASED" if vr["verdict"] == "BIASED" else "FAIR"
                        st.markdown(
                            f"""
                            <div style="
                                background:{vr["bg"]};
                                border:2px solid {vr["border"]};
                                border-radius:10px;
                                padding:14px 12px;
                                margin-bottom:8px;
                                text-align:center;
                            ">
                                <div style="font-size:11px;color:#aaa;margin-bottom:4px;">
                                    {vr["attr"].upper()}
                                </div>
                                <div style="font-size:16px;font-weight:bold;color:#fff;margin-bottom:6px;">
                                    {vr["group"]}
                                </div>
                                <div style="font-size:22px;font-weight:bold;color:{vr["border"]};margin-bottom:6px;">
                                    {label}
                                </div>
                                <div style="font-size:11px;color:#ccc;">
                                    {vr["reason"]}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

            # Summary counts
            total  = len(verdict_rows)
            biased = sum(1 for v in verdict_rows if v["verdict"] == "BIASED")
            fair   = total - biased

            st.markdown("---")
            s1, s2, s3 = st.columns(3)
            s1.metric("Total Groups Audited", total)
            s2.metric("Fair Groups",   fair,   delta=f"{fair/total*100:.0f}%")
            s3.metric("Biased Groups", biased, delta=f"{biased/total*100:.0f}%",
                      delta_color="inverse")

            # Key findings
            st.markdown("#### 🔍 Key Findings")
            col1, col2 = st.columns(2)
            with col1:
                st.error("**Geographic Bias**\n\n"
                         "- Africa PPR = 0.000 — never predicted as high readiness\n"
                         "- Europe FPR = 0.727 — over-predicted 72.7% of the time")
            with col2:
                st.error("**Income Bias**\n\n"
                         "- High income PPR = 0.837 vs Lower middle PPR = 0.100\n"
                         "- 8× disparity between richest and poorest income groups")

        else:
            st.info("Fairness summary available but expected metric columns (ppr, fpr, fnr) not found.")

    except Exception as e:
        st.warning(f"Fairness results not available. Run `05_fairness_audit.py` first.\n\nError: {e}")