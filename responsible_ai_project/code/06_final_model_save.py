# -*- coding: utf-8 -*-
"""
06_final_model_save.py
-----------------------
Trains the final LSTM model on all countries' synthetic time series and saves:
  - model/lstm_model.h5           -> final production model
  - checkpoint/lstm_checkpoint.h5 -> best weights via ModelCheckpoint callback
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH       = 'data/AI_index_db.csv'
MODEL_PATH      = 'model/lstm_model.h5'
CHECKPOINT_PATH = 'checkpoint/lstm_checkpoint.h5'
LOOKBACK        = 4
EPOCHS          = 50
np.random.seed(42)

os.makedirs('model',      exist_ok=True)
os.makedirs('checkpoint', exist_ok=True)

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
print(f"Synthetic series: {len(all_ts)} rows, {all_ts['country'].nunique()} countries.")

# ── 3. Prepare training sequences ─────────────────────────────────────────────
scaler     = MinMaxScaler()
all_values = all_ts['y'].values.reshape(-1, 1)
all_scaled = scaler.fit_transform(all_values)

def create_sequences(data, lookback):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i])
        y.append(data[i])
    return np.array(X), np.array(y)

X, y     = create_sequences(all_scaled, LOOKBACK)
split    = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
print(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}")

# ── 4. Build LSTM model ───────────────────────────────────────────────────────
model = Sequential([
    LSTM(50, activation='relu', input_shape=(LOOKBACK, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.summary()

# ── 5. Train with checkpoint ──────────────────────────────────────────────────
checkpoint_cb = ModelCheckpoint(
    filepath=CHECKPOINT_PATH,
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    validation_split=0.1,
    callbacks=[checkpoint_cb],
    verbose=1
)

# ── 6. Evaluate ───────────────────────────────────────────────────────────────
pred_scaled   = model.predict(X_test, verbose=0)
pred          = scaler.inverse_transform(pred_scaled)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
rmse = np.sqrt(mean_squared_error(y_test_actual, pred))
print(f"\nFinal LSTM Test RMSE: {rmse:.4f}")

# ── 7. Save final model ───────────────────────────────────────────────────────
model.save(MODEL_PATH)
print(f"\nFinal model saved   -> {MODEL_PATH}")
print(f"Best checkpoint     -> {CHECKPOINT_PATH}")
