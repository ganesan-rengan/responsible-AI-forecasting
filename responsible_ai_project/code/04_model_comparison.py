
"""
04_model_comparison.py
-----------------------
Compares ARIMA, SARIMA, SARIMAX, Prophet and LSTM across all countries
using synthetic quarterly time series derived from the AI Index dataset.
All 5 models are evaluated inside a single evaluate_country() function.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH = 'data/AI_index_db.csv'
np.random.seed(42)

# ── 1. Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} countries.")

# ── 2. Generate synthetic quarterly time series ───────────────────────────────
def generate_synthetic_ts(country_name, score_2024):
    dates = pd.date_range(start='2015-01-01', periods=40, freq='Q')
    n     = len(dates)
    delta = np.random.uniform(5, 15)
    start = max(0, score_2024 - delta)
    trend    = np.linspace(start, score_2024, n)
    seasonal = 0.03 * score_2024 * np.sin(2 * np.pi * np.arange(n) / 4)
    noise    = np.random.normal(0, 0.01 * score_2024, n)
    y = np.maximum(trend + seasonal + noise, 0)
    return pd.DataFrame({'ds': dates, 'y': y, 'country': country_name})

all_ts = pd.concat([
    generate_synthetic_ts(row['Country'], row['Total score'])
    for _, row in df.iterrows()
], ignore_index=True)
print(f"Generated {len(all_ts)} rows for {all_ts['country'].nunique()} countries.")

# ── 3. Evaluate ALL 5 models per country ─────────────────────────────────────
def evaluate_country(country_name, country_data):
    train   = country_data.iloc[:-4].copy()
    test    = country_data.iloc[-4:].copy()
    y_train = train['y'].values
    y_test  = test['y'].values
    results = {'country': country_name}
    order, seasonal_order = None, None

    # ── ARIMA ─────────────────────────────────────────────────────────────────
    try:
        m  = auto_arima(y_train, seasonal=False, trace=False,
                        error_action='ignore', suppress_warnings=True)
        fc = m.predict(n_periods=4)
        results['ARIMA_MAE']  = mean_absolute_error(y_test, fc)
        results['ARIMA_RMSE'] = np.sqrt(mean_squared_error(y_test, fc))
    except Exception as e:
        results['ARIMA_MAE'] = results['ARIMA_RMSE'] = np.nan
        print(f"ARIMA failed for {country_name}: {e}")

    # ── SARIMA ────────────────────────────────────────────────────────────────
    try:
        m  = auto_arima(y_train, seasonal=True, m=4, trace=False,
                        error_action='ignore', suppress_warnings=True)
        fc = m.predict(n_periods=4)
        results['SARIMA_MAE']  = mean_absolute_error(y_test, fc)
        results['SARIMA_RMSE'] = np.sqrt(mean_squared_error(y_test, fc))
        order, seasonal_order  = m.order, m.seasonal_order
    except Exception as e:
        results['SARIMA_MAE'] = results['SARIMA_RMSE'] = np.nan
        print(f"SARIMA failed for {country_name}: {e}")

    # ── SARIMAX ───────────────────────────────────────────────────────────────
    if order is not None:
        try:
            np.random.seed(abs(hash(country_name)) % 2**32)
            exog = np.random.normal(0, 1, len(country_data)).cumsum()
            exog = (exog - exog.min()) / (exog.max() - exog.min()) * 10
            sxm  = SARIMAX(y_train, exog=exog[:-4].reshape(-1, 1),
                           order=order, seasonal_order=seasonal_order,
                           enforce_stationarity=False, enforce_invertibility=False)
            sxr  = sxm.fit(disp=False)
            fc   = sxr.forecast(steps=4, exog=exog[-4:].reshape(-1, 1))
            results['SARIMAX_MAE']  = mean_absolute_error(y_test, fc)
            results['SARIMAX_RMSE'] = np.sqrt(mean_squared_error(y_test, fc))
        except Exception as e:
            results['SARIMAX_MAE'] = results['SARIMAX_RMSE'] = np.nan
            print(f"SARIMAX failed for {country_name}: {e}")
    else:
        results['SARIMAX_MAE'] = results['SARIMAX_RMSE'] = np.nan

    # ── Prophet ───────────────────────────────────────────────────────────────
    try:
        pm_model = Prophet(yearly_seasonality=False, weekly_seasonality=False,
                           daily_seasonality=False)
        pm_model.add_seasonality(name='quarterly', period=91.25, fourier_order=3)
        pm_model.fit(train[['ds', 'y']])
        future   = pm_model.make_future_dataframe(periods=4, freq='QE')
        forecast = pm_model.predict(future)
        fc = forecast.set_index('ds').reindex(test['ds'])['yhat'].values
        results['Prophet_MAE']  = mean_absolute_error(y_test, fc)
        results['Prophet_RMSE'] = np.sqrt(mean_squared_error(y_test, fc))
    except Exception as e:
        results['Prophet_MAE'] = results['Prophet_RMSE'] = np.nan
        print(f"Prophet failed for {country_name}: {e}")

    # ── LSTM ──────────────────────────────────────────────────────────────────
    try:
        lookback = 4
        values   = country_data['y'].values.reshape(-1, 1)
        scaler   = MinMaxScaler()
        scaled   = scaler.fit_transform(values)
        split    = int(len(scaled) * 0.8)
        tr_s, te_s = scaled[:split], scaled[split:]

        def make_seq(data, lb):
            X, y = [], []
            for i in range(lb, len(data)):
                X.append(data[i - lb:i])
                y.append(data[i])
            return np.array(X), np.array(y)

        X_tr, y_tr = make_seq(tr_s, lookback)
        X_te, y_te = make_seq(te_s, lookback)

        if len(X_tr) == 0 or len(X_te) == 0:
            raise ValueError("Not enough sequences for LSTM")

        lstm_model = Sequential([
            LSTM(50, activation='relu', input_shape=(lookback, 1)),
            Dense(1)
        ])
        lstm_model.compile(optimizer='adam', loss='mse')
        lstm_model.fit(X_tr, y_tr, epochs=50, verbose=0, validation_split=0.1)

        pred   = scaler.inverse_transform(lstm_model.predict(X_te, verbose=0))
        actual = scaler.inverse_transform(y_te.reshape(-1, 1))

        results['LSTM_MAE']  = mean_absolute_error(actual, pred)
        results['LSTM_RMSE'] = np.sqrt(mean_squared_error(actual, pred))
    except Exception as e:
        results['LSTM_MAE'] = results['LSTM_RMSE'] = np.nan
        print(f"LSTM failed for {country_name}: {e}")

    return results

# ── 4. Run evaluation for all countries ───────────────────────────────────────
all_results = []
for country in tqdm(all_ts['country'].unique(), desc="Evaluating all models"):
    country_data = all_ts[all_ts['country'] == country].sort_values('ds')
    all_results.append(evaluate_country(country, country_data))

comparison = pd.DataFrame(all_results)

# ── 5. Print summary ──────────────────────────────────────────────────────────
mae_cols  = ['ARIMA_MAE',  'SARIMA_MAE',  'SARIMAX_MAE',  'Prophet_MAE',  'LSTM_MAE']
rmse_cols = ['ARIMA_RMSE', 'SARIMA_RMSE', 'SARIMAX_RMSE', 'Prophet_RMSE', 'LSTM_RMSE']

print("\nAverage MAE across countries:")
print(comparison[mae_cols].mean())
print("\nAverage RMSE across countries:")
print(comparison[rmse_cols].mean())
print("\nBest model (by MAE) counts:")
print(comparison[mae_cols].idxmin(axis=1).value_counts())
print("\nBest model (by RMSE) counts:")
print(comparison[rmse_cols].idxmin(axis=1).value_counts())

# ── 6. Plot – styled bar charts (RMSE + MAE) ──────────────────────────────────
os.makedirs('data/plots', exist_ok=True)

model_labels  = ['ARIMA', 'SARIMA', 'SARIMAX', 'Prophet', 'LSTM']
rmse_vals     = comparison[rmse_cols].mean().values
mae_vals      = comparison[mae_cols].mean().values
best_rmse_idx = int(np.nanargmin(rmse_vals))
best_mae_idx  = int(np.nanargmin(mae_vals))

palette         = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2']
bar_colors_rmse = ['#FFD700' if i == best_rmse_idx else palette[i] for i in range(5)]
bar_colors_mae  = ['#FFD700' if i == best_mae_idx  else palette[i] for i in range(5)]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Model Comparison – Average Error Across All Countries',
             fontsize=15, fontweight='bold', y=1.02)

# RMSE chart
bars = axes[0].bar(model_labels, rmse_vals, color=bar_colors_rmse,
                   edgecolor='black', linewidth=0.8, width=0.55)
axes[0].set_title('Average RMSE by Model', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Model', fontsize=11)
axes[0].set_ylabel('RMSE', fontsize=11)
axes[0].set_ylim(0, np.nanmax(rmse_vals) * 1.25)
axes[0].grid(axis='y', linestyle='--', alpha=0.6)
axes[0].set_axisbelow(True)
for bar, val in zip(bars, rmse_vals):
    if not np.isnan(val):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + np.nanmax(rmse_vals) * 0.02,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
axes[0].annotate(f'Best: {model_labels[best_rmse_idx]}',
                 xy=(best_rmse_idx, rmse_vals[best_rmse_idx]),
                 xytext=(best_rmse_idx, rmse_vals[best_rmse_idx] + np.nanmax(rmse_vals) * 0.12),
                 ha='center', fontsize=9, color='darkgreen', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1.5))

# MAE chart
bars2 = axes[1].bar(model_labels, mae_vals, color=bar_colors_mae,
                    edgecolor='black', linewidth=0.8, width=0.55)
axes[1].set_title('Average MAE by Model', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Model', fontsize=11)
axes[1].set_ylabel('MAE', fontsize=11)
axes[1].set_ylim(0, np.nanmax(mae_vals) * 1.25)
axes[1].grid(axis='y', linestyle='--', alpha=0.6)
axes[1].set_axisbelow(True)
for bar, val in zip(bars2, mae_vals):
    if not np.isnan(val):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + np.nanmax(mae_vals) * 0.02,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
axes[1].annotate(f'Best: {model_labels[best_mae_idx]}',
                 xy=(best_mae_idx, mae_vals[best_mae_idx]),
                 xytext=(best_mae_idx, mae_vals[best_mae_idx] + np.nanmax(mae_vals) * 0.12),
                 ha='center', fontsize=9, color='darkgreen', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1.5))

plt.tight_layout()
plt.savefig('data/plots/model_comparison.png', dpi=150, bbox_inches='tight')
print("Plot saved → data/plots/model_comparison.png")
plt.show()

# ── 7. Save results ───────────────────────────────────────────────────────────
comparison.to_csv('data/model_comparison_results.csv', index=False)
print("\nResults saved → data/model_comparison_results.csv")