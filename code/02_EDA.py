# -*- coding: utf-8 -*-
"""
02_EDA.py
----------
Exploratory Data Analysis on the AI Index dataset.
- AutoViz automated visualisation
- Stationarity test (ADF)
- ACF / PACF plots (original, 1st and 2nd differencing)
- Seasonal vs Usual differencing comparison
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from autoviz.AutoViz_Class import AutoViz_Class
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH  = 'data/AI_index_db.csv'
START_DATE = '2000-01-01'

# ── 1. Load and prepare data ──────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
dates = pd.date_range(start=START_DATE, periods=len(df), freq='D')
df['date'] = dates
df = df.set_index('date')
df['value'] = df['Total score']
print(f"Loaded {len(df)} rows.")

# ── 2. AutoViz automated EDA ──────────────────────────────────────────────────
print("\nRunning AutoViz...")
Av = AutoViz_Class()
df_autoviz = Av.AutoViz(DATA_PATH)

# ── 3. ADF stationarity test ──────────────────────────────────────────────────
print("\n--- ADF Stationarity Test ---")
result = adfuller(df['value'].dropna())
print(f"ADF Statistic : {result[0]:.4f}")
print(f"p-value       : {result[1]:.4f}")
if result[1] < 0.05:
    print("  -> Series is stationary (reject H0)")
else:
    print("  -> Series is NON-stationary (fail to reject H0)")

# ── 4. ACF / PACF plots ───────────────────────────────────────────────────────
plt.rcParams.update({'figure.figsize': (9, 7), 'figure.dpi': 120})

fig, axes = plt.subplots(3, 2, sharex=True)
axes[0, 0].plot(df['value'])
axes[0, 0].set_title('Original Series')
plot_acf(df['value'], ax=axes[0, 1])

axes[1, 0].plot(df['value'].diff())
axes[1, 0].set_title('1st Order Differencing')
plot_acf(df['value'].diff().dropna(), ax=axes[1, 1])

axes[2, 0].plot(df['value'].diff().diff())
axes[2, 0].set_title('2nd Order Differencing')
plot_acf(df['value'].diff().diff().dropna(), ax=axes[2, 1])
plt.tight_layout()
plt.show()

# PACF of 1st differenced series
plt.rcParams.update({'figure.figsize': (9, 3), 'figure.dpi': 120})
fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(df['value'].diff())
axes[0].set_title('1st Differencing')
axes[1].set(ylim=(0, 5))
plot_pacf(df['value'].diff().dropna(), ax=axes[1])
plt.show()

# ACF of 1st differenced series
fig, axes = plt.subplots(1, 2, sharex=True)
axes[0].plot(df['value'].diff())
axes[0].set_title('1st Differencing')
axes[1].set(ylim=(0, 1.2))
plot_acf(df['value'].diff().dropna(), ax=axes[1])
plt.show()

# ── 5. Seasonal vs Usual differencing ────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(10, 5), dpi=100, sharex=True)
axes[0].plot(df['value'], label='Original Series')
axes[0].plot(df['value'].diff(1), label='Usual Differencing')
axes[0].set_title('Usual Differencing')
axes[0].legend(loc='upper left', fontsize=10)

axes[1].plot(df['value'], label='Original Series')
axes[1].plot(df['value'].diff(12), label='Seasonal Differencing', color='green')
axes[1].set_title('Seasonal Differencing')
plt.legend(loc='upper left', fontsize=10)
plt.suptitle('AI Index - Time Series Dataset', fontsize=16)
plt.tight_layout()
plt.show()

print("\nEDA complete.")
