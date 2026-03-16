# -*- coding: utf-8 -*-
"""
05_fairness_audit.py
---------------------
Fairness audit using the Aequitas library.
Converts regression outputs to binary (median threshold) then checks
disparity metrics across world regions using 'Americas' as reference group.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from aequitas.group import Group
from aequitas.bias import Bias
from aequitas.fairness import Fairness

# ── Ensure results folder exists ─────────────────────────────────────────────
os.makedirs("results", exist_ok=True)

# ── Load comparison results produced by 04_model_comparison.py ───────────────
DATA_PATH    = 'data/AI_index_db.csv'
RESULTS_PATH = 'data/model_comparison_results.csv'

df_raw     = pd.read_csv(DATA_PATH)
results_df = pd.read_csv(RESULTS_PATH)

# ── Build y_test_actual, y_pred and test_regions from data ───────────────────
country_region = df_raw.set_index('Country')['Region'].to_dict()
results_df['region'] = results_df['country'].map(country_region)
results_df = results_df.dropna(subset=['region'])

# Proxy prediction construction
country_score  = df_raw.set_index('Country')['Total score'].to_dict()
results_df['actual']    = results_df['country'].map(country_score)
results_df['predicted'] = results_df['actual'] - results_df['ARIMA_RMSE'].fillna(0)

y_test_actual = results_df['actual'].values
y_pred        = results_df['predicted'].values
test_regions = results_df['region'].values


test_regions = results_df['region'].values

# ── 1. Convert to binary using median threshold ───────────────────────────────
test_regions = np.array([str(r).strip() for r in test_regions])
threshold    = np.median(y_test_actual)

actual_class = (y_test_actual > threshold).astype(int)
pred_class   = (y_pred        > threshold).astype(int)

aeq_df = pd.DataFrame({
    'score':       pred_class.flatten(),
    'label_value': actual_class.flatten(),
    'region':      test_regions
})

print("Unique regions:", aeq_df['region'].unique())

# ── 2. Group metrics ──────────────────────────────────────────────────────────
g = Group()
xtab, _ = g.get_crosstabs(aeq_df)

# ── 3. Disparity metrics ──────────────────────────────────────────────────────
desired_ref = 'Americas'

if desired_ref not in aeq_df['region'].unique():
    desired_ref = aeq_df['region'].mode()[0]
    print(f"Reference group adjusted to: '{desired_ref}'")

b   = Bias()
bdf = b.get_disparity_predefined_groups(
    xtab,
    original_df=aeq_df,
    ref_groups_dict={'region': desired_ref},
    alpha=0.05
)

# ── 4. Fairness check ─────────────────────────────────────────────────────────
f   = Fairness()
fdf = f.get_group_value_fairness(bdf)

# ── 5. Display results ────────────────────────────────────────────────────────
print("\nFairness audit results:")
print("Available columns:", fdf.columns.tolist())

desired_cols   = ['attribute_name', 'attribute_value', 'ppr', 'fpr', 'fnr',
                  'ppr_is_fair', 'fpr_is_fair', 'fnr_is_fair']

available_cols = [c for c in desired_cols if c in fdf.columns]

if not available_cols:
    print(fdf.to_string())
else:
    print(fdf[available_cols].to_string())

# ── Visualization for README ──────────────────────────────────────────────
try:
    metric_col = None

    # find correct fairness metric column
    for col in ['ppr', 'ppr_disparity', 'fpr', 'fnr']:
        if col in fdf.columns:
            metric_col = col
            break

    if metric_col is None:
        print("No fairness metric column found for visualization.")
    else:
        plot_df = fdf[['attribute_value', metric_col]].dropna()

        plt.figure(figsize=(10,6))
        sns.barplot(data=plot_df, x='attribute_value', y=metric_col, palette='viridis')

        plt.title(f"Fairness Audit by Region ({metric_col.upper()} Metric)")
        plt.xlabel("Region")
        plt.ylabel(metric_col.upper())

        plt.xticks(rotation=45)
        plt.grid(True)

        plt.savefig("results/fairness_audit.png", bbox_inches="tight")
        plt.show()

        print(f"\nFairness visualization saved using column: {metric_col}")

except Exception as e:
    print("\nVisualization skipped:", e)

# ── 6. Fairness Summary Table ───────────────────────────────────────────────
print("\nFairness Summary Table:")

summary_cols = ['attribute_value', 'ppr', 'fpr', 'fnr']

available_summary_cols = [c for c in summary_cols if c in fdf.columns]

if available_summary_cols:
    fairness_summary = fdf[available_summary_cols].drop_duplicates()
    print(fairness_summary.to_string(index=False))
else:
    print("Required fairness metrics not available in results.")    


fairness_summary.to_csv("results/fairness_summary.csv", index=False)    