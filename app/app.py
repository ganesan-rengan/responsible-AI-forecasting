# -*- coding: utf-8 -*-
"""
app/app.py
-----------
Streamlit dashboard for the Responsible AI project.
Select a country to view its synthetic time series and LSTM forecast.
Run with: streamlit run app/app.py
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ── Config ─────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'AI_index_db.csv')
RESULTS_FAIRNESS = os.path.join(BASE_DIR, '..', 'results', 'fairness_summary.csv')

LOOKBACK  = 4
EPOCHS    = 30
np.random.seed(42)

st.set_page_config(page_title="Responsible AI - AI Index Forecaster", layout="wide")
st.title("Responsible AI - AI Index Forecaster")

# ── 1. Load data ───────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

st.subheader("Raw Dataset Preview")
st.dataframe(df.head(20), use_container_width=True)

# ── 2. Generate synthetic time series ──────────────────────────────
@st.cache_data
def build_all_ts(df):

    def generate_synthetic_ts(country_name, score_2024):

        dates = pd.date_range(start='2015-01-01', periods=40, freq='Q')
        n = len(dates)

        delta = np.random.uniform(5, 15)
        start = max(0, score_2024 - delta)

        trend = np.linspace(start, score_2024, n)
        seasonal = 0.03 * score_2024 * np.sin(2 * np.pi * np.arange(n) / 4)
        noise = np.random.normal(0, 0.01 * score_2024, n)

        y = np.maximum(trend + seasonal + noise, 0)

        return pd.DataFrame({
            'ds': dates,
            'y': y,
            'country': country_name
        })

    return pd.concat([
        generate_synthetic_ts(row['Country'], row['Total score'])
        for _, row in df.iterrows()
    ], ignore_index=True)

all_ts = build_all_ts(df)

# ── 3. Country selector & forecast ─────────────────────────────────
st.subheader("Country Forecast")

country = st.selectbox("Select a country", sorted(df['Country'].unique()))

country_data = all_ts[all_ts['country'] == country].sort_values('ds').reset_index(drop=True)

values = country_data['y'].values.reshape(-1, 1)

scaler = MinMaxScaler()
scaled = scaler.fit_transform(values)

split = int(len(scaled) * 0.8)

train_s = scaled[:split]
test_s  = scaled[split:]

def make_seq(data, lb):
    X, y = [], []
    for i in range(lb, len(data)):
        X.append(data[i - lb:i])
        y.append(data[i])
    return np.array(X), np.array(y)

X_tr, y_tr = make_seq(train_s, LOOKBACK)
X_te, y_te = make_seq(test_s, LOOKBACK)

if len(X_tr) > 0 and len(X_te) > 0:

    with st.spinner("Training LSTM model..."):

        model = Sequential([
            LSTM(50, activation='relu', input_shape=(LOOKBACK, 1)),
            Dense(1)
        ])

        model.compile(optimizer='adam', loss='mse')

        model.fit(
            X_tr,
            y_tr,
            epochs=EPOCHS,
            verbose=0,
            validation_split=0.1
        )

    pred = scaler.inverse_transform(model.predict(X_te, verbose=0))
    actual_test = scaler.inverse_transform(y_te.reshape(-1, 1))

    rmse = np.sqrt(mean_squared_error(actual_test, pred))

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(country_data['ds'].values, values, label='Full Series', color='steelblue')

    test_dates = country_data['ds'].values[split + LOOKBACK:]

    ax.plot(test_dates, actual_test, label='Actual (test)', color='orange')
    ax.plot(test_dates, pred, label='LSTM Forecast', color='red', linestyle='--')

    ax.set_title(f'{country} - LSTM Forecast (RMSE: {rmse:.2f})')

    ax.legend()

    st.pyplot(fig)

    st.metric("Test RMSE", f"{rmse:.4f}")

else:
    st.warning("Not enough data to train LSTM for this country.")

# ── 4. Responsible AI Fairness Evaluation ──────────────────────────
st.subheader("Responsible AI Fairness Evaluation")

try:

    fairness_df = pd.read_csv(RESULTS_FAIRNESS)

    st.write("Fairness Metrics by Group")
    st.dataframe(fairness_df, use_container_width=True)

    if {'attribute_value','ppr','fpr','fnr'}.issubset(fairness_df.columns):

        fig2, ax2 = plt.subplots(figsize=(8,4))

        fairness_df.set_index("attribute_value")[["ppr","fpr","fnr"]].plot(
            kind="bar",
            ax=ax2
        )

        ax2.set_title("Fairness Metrics Comparison")
        ax2.set_xlabel("Group")
        ax2.set_ylabel("Metric Value")

        st.pyplot(fig2)

    else:
        st.info("Fairness summary available but expected metric columns not found.")

except Exception:
    st.warning("Fairness results not available. Run fairness audit script first.")

st.caption("Data: AI Readiness Index | Model: LSTM | Fairness: Aequitas")