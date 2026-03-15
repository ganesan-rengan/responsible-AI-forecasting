# -*- coding: utf-8 -*-
"""
01_data_preprocessing.py
-------------------------
Loads and cleans the AI Index dataset.
- Fixes data quality issues using FixDQ
- Preserves 'Total score' before FixDQ drops it
- Creates a synthetic date index (data lacks timestamps)
- Assigns 'Total score' to a new 'value' column for time series compatibility
"""

import os
import numpy as np
import pandas as pd
from autoviz import FixDQ

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH  = 'data/AI_index_db.csv'
START_DATE = '2000-01-01'

# ── 1. Load raw data ──────────────────────────────────────────────────────────
print("Loading dataset...")
df_raw = pd.read_csv(DATA_PATH)
print(f"  Rows: {len(df_raw)}, Columns: {len(df_raw.columns)}")

# ── 2. Preserve 'Total score' BEFORE FixDQ drops it ──────────────────────────
# FixDQ drops 'Total score' because it is highly correlated with 'Research'
total_score = df_raw['Total score'].copy()
country_col = df_raw['Country'].copy() if 'Country' in df_raw.columns else None

# ── 3. Fix data quality issues ────────────────────────────────────────────────
print("\nFixing data quality issues with FixDQ...")
fixdq    = FixDQ()
df_clean = fixdq.fit_transform(df_raw)
print("Cleaned data sample:")
print(df_clean.head())

# ── 4. Restore dropped columns ────────────────────────────────────────────────
df_clean['Total score'] = total_score.values
if country_col is not None:
    df_clean['Country'] = country_col.values

# ── 5. Create synthetic date index ────────────────────────────────────────────
print("\nCreating synthetic date index...")
dates = pd.date_range(start=START_DATE, periods=len(df_clean), freq='D')
df_clean['date'] = dates
df_clean = df_clean.set_index('date')

# ── 6. Create 'value' column for time series ──────────────────────────────────
df_clean['value'] = df_clean['Total score']

print("\nFinal preprocessed data:")
print(df_clean.head())

# ── 7. Save cleaned data ──────────────────────────────────────────────────────
df_clean.to_csv('data/AI_index_db_clean.csv')
print("\nCleaned data saved → data/AI_index_db_clean.csv")
