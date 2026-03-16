# -*- coding: utf-8 -*-
"""
03_models_training.py
----------------------
Trains time-series models on the AI Index dataset:
  - ARIMA  (manual + auto via pmdarima)
  - SARIMA (auto via pmdarima)
  - SARIMAX (seasonal exogenous variable)
  - Prophet (per-country synthetic series)
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf

import pmdarima as pm
from prophet import Prophet

# ── Create results folder ─────────────────────────────────────────────────────
os.makedirs("results", exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH  = 'data/AI_index_db.csv'
START_DATE = '2000-01-01'
N_FORECAST = 24
np.random.seed(42)

# ── 1. Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
dates = pd.date_range(start=START_DATE, periods=len(df), freq='D')
df['date'] = dates
df = df.set_index('date')
df['value'] = df['Total score']
print(f"Loaded {len(df)} rows.")

# ── 2. Manual ARIMA ───────────────────────────────────────────────────────────
print("\n--- Manual ARIMA (1,1,1) ---")
train = df['value'][:-12]
test  = df['value'][-12:]

model_arima  = ARIMA(train, order=(1, 1, 1))
fitted_arima = model_arima.fit()
print(fitted_arima.summary())

# Residuals plot
residuals = pd.DataFrame(fitted_arima.resid)
fig, ax = plt.subplots(1, 2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.tight_layout()
plt.show()

# Actual vs Fitted
fitted_values = fitted_arima.predict(dynamic=False)
plt.figure(figsize=(10, 6))
plt.plot(df['value'], label='Actual Values')
plt.plot(fitted_values, label='Fitted Values', color='red')
plt.title('Actual vs Fitted Values')
plt.legend()
plt.show()

# Forecast
forecast_results = fitted_arima.get_forecast(steps=len(test))
fc           = forecast_results.predicted_mean
conf         = forecast_results.conf_int(alpha=0.05)
fc_series    = pd.Series(fc.values,              index=test.index)
lower_series = pd.Series(conf.iloc[:, 0].values, index=test.index)
upper_series = pd.Series(conf.iloc[:, 1].values, index=test.index)

plt.figure(figsize=(12, 5), dpi=100)
plt.plot(train,        label='Training')
plt.plot(test,         label='Actual')
plt.plot(fc_series,    label='Forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, color='k', alpha=.15)
plt.title('ARIMA Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)

# 🔹 Save forecast image for README
plt.savefig("results/forecast_plot.png", bbox_inches="tight")

plt.show()

# Accuracy metrics
def forecast_accuracy(forecast, actual):
    mape   = np.mean(np.abs(forecast - actual) / np.abs(actual))
    me     = np.mean(forecast - actual)
    mae    = np.mean(np.abs(forecast - actual))
    mpe    = np.mean((forecast - actual) / actual)
    rmse   = np.mean((forecast - actual) ** 2) ** .5
    corr   = np.corrcoef(forecast, actual)[0, 1]
    mins   = np.amin(np.hstack([forecast.values[:, None], actual.values[:, None]]), axis=1)
    maxs   = np.amax(np.hstack([forecast.values[:, None], actual.values[:, None]]), axis=1)
    minmax = 1 - np.mean(mins / maxs)
    acf1   = acf(forecast - actual)[1]
    return dict(mape=mape, me=me, mae=mae, mpe=mpe,
                rmse=rmse, acf1=acf1, corr=corr, minmax=minmax)

print("\nForecast accuracy:", forecast_accuracy(fc_series, test))

# ── 3. Auto-ARIMA ─────────────────────────────────────────────────────────────
print("\n--- Auto ARIMA ---")
auto_model = pm.auto_arima(df['value'], start_p=1, start_q=1,
                           test='adf', max_p=3, max_q=3, m=1,
                           d=None, seasonal=False, start_P=0, D=0,
                           trace=True, error_action='ignore',
                           suppress_warnings=True, stepwise=True)
print(auto_model.summary())

fc_auto, confint_auto = auto_model.predict(n_periods=N_FORECAST, return_conf_int=True)
idx = np.arange(len(df['value']), len(df['value']) + N_FORECAST)
plt.figure(figsize=(10, 4))
plt.plot(df['value'].values, label='Historical')
plt.plot(idx, fc_auto, color='darkgreen', label='Forecast')
plt.fill_between(idx, confint_auto[:, 0], confint_auto[:, 1], color='k', alpha=.15)
plt.title("Auto-ARIMA Final Forecast")
plt.legend()
plt.show()

# ── 4. SARIMA ─────────────────────────────────────────────────────────────────
print("\n--- SARIMA ---")
smodel = pm.auto_arima(df['value'], start_p=1, start_q=1,
                       test='adf', max_p=3, max_q=3, m=12,
                       start_P=0, seasonal=True, d=None, D=1,
                       trace=True, error_action='ignore',
                       suppress_warnings=True, stepwise=True)
print(smodel.summary())
smodel.plot_diagnostics(figsize=(10, 8))
plt.show()

fitted_s, confint_s = smodel.predict(n_periods=N_FORECAST, return_conf_int=True)
index_of_fc = pd.date_range(df.index[-1], periods=N_FORECAST, freq='MS')
plt.figure(figsize=(10, 4))
plt.plot(df['value'], label='Historical')
plt.plot(pd.Series(fitted_s, index=index_of_fc), color='darkgreen', label='Forecast')
plt.fill_between(index_of_fc, confint_s[:, 0], confint_s[:, 1], color='k', alpha=.15)
plt.title("SARIMA Final Forecast")
plt.legend()
plt.show()

# ── 5. SARIMAX ────────────────────────────────────────────────────────────────
print("\n--- SARIMAX ---")
result_mul = seasonal_decompose(df['value'][-36:], model='additive', extrapolate_trend='freq')
seas_df = result_mul.seasonal.to_frame()
seas_df.columns = ['seasonal_component']
seas_df['month'] = pd.to_datetime(seas_df.index).month
seasonal_factors_map = seas_df.groupby('month')['seasonal_component'].mean()

sarimax_df = df[['value']].copy()
sarimax_df['month'] = df.index.month
sarimax_df['seasonal_index'] = sarimax_df['month'].map(seasonal_factors_map)

sxmodel = pm.auto_arima(sarimax_df[['value']],
                        exogenous=sarimax_df[['seasonal_index']],
                        start_p=1, start_q=1, test='adf',
                        max_p=3, max_q=3, m=12,
                        start_P=0, seasonal=True, d=None, D=1,
                        trace=True, error_action='ignore',
                        suppress_warnings=True, stepwise=True)
print(sxmodel.summary())

# ── 6. Prophet – per-country synthetic series ─────────────────────────────────
print("\n--- Prophet ---")
raw_df  = pd.read_csv(DATA_PATH)
records = []

for country in raw_df['Country'].unique():
    score_2024 = raw_df.loc[raw_df['Country'] == country, 'Total score'].values[0]
    years  = list(range(2015, 2025))
    base   = max(0, score_2024 - np.random.uniform(5, 15))
    trend  = np.linspace(base, score_2024, len(years))
    noise  = np.random.normal(0, 0.5, len(years))
    scores = np.maximum(trend + noise, 0)
    for yr, score in zip(years, scores):
        records.append({'ds': pd.Timestamp(f'{yr}-12-31'), 'y': score, 'country': country})

ts_df = pd.DataFrame(records)
print(f"Synthetic time series created for {ts_df['country'].nunique()} countries.")

forecasts = {}
for country in ts_df['country'].unique():
    country_data = ts_df[ts_df['country'] == country][['ds', 'y']].copy()
    if len(country_data) < 2:
        continue
    m = Prophet(yearly_seasonality=False, daily_seasonality=False, weekly_seasonality=False)
    m.fit(country_data)
    future   = m.make_future_dataframe(periods=2, freq='YE')
    forecast = m.predict(future)
    forecasts[country] = forecast

print(f"Prophet models fitted for {len(forecasts)} countries.")

plt.figure(figsize=(12, 8))
for country, forecast in forecasts.items():
    future_fc = forecast[forecast['ds'] > pd.Timestamp('2024-12-31')]
    plt.plot(future_fc['ds'], future_fc['yhat'], label=country, alpha=0.6)

plt.title('Prophet Forecasts for All Countries (2025-2026)')
plt.xlabel('Year')
plt.ylabel('Total Score')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
plt.grid(True)
plt.tight_layout()
plt.show()

print("\nModels training complete.")