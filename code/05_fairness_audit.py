# -*- coding: utf-8 -*-
"""
05_fairness_audit.py
FINAL VERSION: Fairness + Classification Metrics + Confusion Matrix
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from aequitas.group import Group
from aequitas.bias import Bias
from aequitas.fairness import Fairness

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ── Config ─────────────────────────────────────────────
DATA_PATH   = 'data/AI_index_db.csv'
RESULTS_DIR = 'results'

os.makedirs(RESULTS_DIR, exist_ok=True)
np.random.seed(42)

print("=" * 60)
print(" RESPONSIBLE AI FAIRNESS AUDIT (FINAL)")
print("=" * 60)

# ── Load Data ──────────────────────────────────────────
df = pd.read_csv(DATA_PATH)

df['Region'] = df['Region'].str.strip()
df['Income group'] = df['Income group'].str.strip()

actual = df['Total score'].values.astype(float)

# ── Simulate Bias (for demonstration) ──────────────────
REGION_BIAS = {
    'Europe': 12,
    'Americas': 2,
    'Asia-Pacific': 1,
    'Middle East': -2,
    'Africa': -18
}

INCOME_BIAS = {
    'High': 10,
    'Upper middle': 1,
    'Lower middle': -15
}

region_offset = df['Region'].map(REGION_BIAS).fillna(0).values
income_offset = df['Income group'].map(INCOME_BIAS).fillna(0).values
noise = np.random.normal(0, 1.5, len(actual))

predicted = np.clip(actual + region_offset + income_offset + noise, 0, 100)

# ── Binary Conversion ──────────────────────────────────
threshold = np.median(actual)

label_value = (actual > threshold).astype(int)
score       = (predicted > threshold).astype(int)

print(f"\nThreshold used: {threshold:.2f}")

# ═══════════════════════════════════════════════════════
# 🔥 CLASSIFICATION METRICS
# ═══════════════════════════════════════════════════════

accuracy  = accuracy_score(label_value, score)
precision = precision_score(label_value, score, zero_division=0)
recall    = recall_score(label_value, score, zero_division=0)
f1        = f1_score(label_value, score, zero_division=0)

print("\nClassification Metrics:")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

# Save metrics
metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
    "Value": [accuracy, precision, recall, f1]
})
metrics_df.to_csv(f"{RESULTS_DIR}/classification_metrics.csv", index=False)

# ═══════════════════════════════════════════════════════
# 🔥 CONFUSION MATRIX (PNG)
# ═══════════════════════════════════════════════════════

cm = confusion_matrix(label_value, score)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Pred 0", "Pred 1"],
            yticklabels=["Actual 0", "Actual 1"])

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.savefig(f"{RESULTS_DIR}/confusion_matrix.png")
plt.close()

print("Saved confusion_matrix.png")

# ═══════════════════════════════════════════════════════
# 🔥 AEQUITAS FAIRNESS
# ═══════════════════════════════════════════════════════

aeq_df = pd.DataFrame({
    'score': score,
    'label_value': label_value,
    'region': df['Region'],
    'income_group': df['Income group']
})

g = Group()
xtab, _ = g.get_crosstabs(aeq_df)

b = Bias()
bdf = b.get_disparity_predefined_groups(
    xtab,
    original_df=aeq_df,
    ref_groups_dict={'region': 'Americas', 'income_group': 'High'},
    alpha=0.05
)

f = Fairness()
fdf = f.get_group_value_fairness(bdf)

print("\nFairness Results:")
print(fdf[['attribute_name','attribute_value','ppr','fpr','fnr']])

# ── Verdict Logic ─────────────────────────────────────
def verdict(row):
    if row['fpr'] > 0.20 or row['ppr'] == 0:
        return "BIASED"
    return "FAIR"

fdf['verdict'] = fdf.apply(verdict, axis=1)

# Save fairness summary
fdf.to_csv(f"{RESULTS_DIR}/fairness_summary.csv", index=False)

print("\nFairness Verdict:")
print(fdf[['attribute_value','ppr','fpr','fnr','verdict']])

print("\n✅ All results saved in /results/")