# Responsible AI Forecasting Framework for E-Governance

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Responsible AI](https://img.shields.io/badge/AI-Fairness-green)
![AEQUITAS](https://img.shields.io/badge/Fairness-AEQUITAS-purple)
![License](https://img.shields.io/badge/License-MIT-yellow)

> **Core Question:** *Are AI-based forecasting models used in governance contexts fair across all regions and income groups — or do they exhibit systematic bias?*

End-to-end **Responsible AI pipeline** for forecasting AI readiness indicators across 62 countries using the **AI Global Index dataset**. The project integrates time-series forecasting, model evaluation, fairness auditing via AEQUITAS, and a deployed Streamlit dashboard.

---

## Project Overview

E-Governance systems increasingly adopt AI to automate public service delivery and policy decision-making. While AI improves efficiency, it may introduce **bias and unfairness** — particularly in how it treats different regions and income groups when forecasting AI readiness.

This project answers the core question through a complete pipeline:

- Data preprocessing and automated EDA
- Five forecasting models trained and compared
- Formal fairness audit using the **AEQUITAS framework**
- Interactive Streamlit dashboard with bias verdict cards

---

## Key Results

| Model   | Avg RMSE | Avg MAE | Verdict              |
|---------|----------|---------|----------------------|
| ARIMA   | 0.66     | 0.59    | Good baseline        |
| SARIMA  | **0.29** | **0.25**| **Best model**       |
| SARIMAX | Failed   | Failed  | Exploded (synthetic exogenous) |
| Prophet | 0.63     | 0.56    | Close second         |
| LSTM    | 2.24     | 2.16    | Acceptable           |

### Fairness Audit Findings (AEQUITAS)

| Group        | PPR   | FPR   | Verdict      |
|--------------|-------|-------|--------------|
| Africa       | 0.000 | 0.000 | BIASED — never predicted positive |
| Europe       | 0.667 | 0.909 | BIASED — massively over-predicted |
| Americas     | 0.071 | 0.167 | FAIR         |
| Asia-Pacific | 0.190 | 0.000 | FAIR         |
| High income  | 0.952 | 0.786 | BIASED — over-predicted |
| Lower middle | 0.000 | 0.000 | BIASED — never predicted positive |

**Conclusion:** The model exhibits significant geographic and income-based bias. Europe is over-predicted at FPR = 0.909, while Africa is never predicted as high-readiness (PPR = 0.000). High income countries are predicted as AI-ready at a rate **8x greater** than Lower middle income countries.

---

## Dataset

**AI Global Index** — Tortoise Media via Kaggle

| Attribute        | Details                                      |
|-----------------|----------------------------------------------|
| Source          | kaggle.com/datasets/katerynameleshenko/ai-index |
| Countries       | 62                                           |
| Features        | 13                                           |
| Target Variable | Total score (0–100)                          |
| Sensitive Attrs | Region, Income group                         |
| Regions         | Americas, Europe, Asia-Pacific, Middle East, Africa |
| Income Groups   | High, Upper middle, Lower middle             |

---

## Pipeline Architecture

```
Raw Dataset (62 countries, 13 features)
        ↓
01_data_preprocessing.py   — AutoViz FixDQ, synthetic date index
        ↓
02_EDA.py                  — AutoViz EDA, ADF test, ACF/PACF
        ↓
03_models_training.py      — ARIMA, SARIMA, SARIMAX, Prophet
        ↓
04_model_comparison.py     — All 5 models × 62 countries (MAE + RMSE)
        ↓
    Best Model: SARIMA (RMSE = 0.29)
        ↓
05_fairness_audit.py       — AEQUITAS fairness audit (Region + Income)
        ↓
06_final_model_save.py     — LSTM saved with ModelCheckpoint
        ↓
app/app.py                 — Streamlit dashboard (forecast + fairness)
```

---

## Project Structure

```
RESPONSIBLE_AI_PROJECT/
│
├── app/
│   └── app.py                    # Streamlit dashboard
│
├── checkpoint/
│   └── lstm_checkpoint.h5        # Best epoch LSTM checkpoint
│
├── code/
│   ├── 01_data_preprocessing.py  # Data loading + FixDQ cleaning
│   ├── 02_EDA.py                 # AutoViz EDA + ADF stationarity
│   ├── 03_models_training.py     # ARIMA, SARIMA, SARIMAX, Prophet
│   ├── 04_model_comparison.py    # 5-model RMSE comparison
│   ├── 05_fairness_audit.py      # AEQUITAS fairness audit
│   └── 06_final_model_save.py    # Final LSTM with ModelCheckpoint
│
├── data/
│   ├── AI_index_db.csv           # Raw dataset
│   └── AI_index_db_clean.csv     # Cleaned dataset
│
├── model/
│   └── lstm_model.h5             # Final trained LSTM weights
│
├── results/
│   ├── fairness_audit.png        # 6-panel AEQUITAS dark theme chart
│   ├── fairness_summary.csv      # PPR, FPR, FNR + verdict per group
│   ├── rmse_comparison.png       # Model comparison bar chart
│   └── forecast_plot.png         # ARIMA forecast vs actuals
│
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Place dataset

```bash
cp /path/to/AI_index_db.csv data/
```

### 3. Run the full pipeline

```bash
python code/01_data_preprocessing.py
python code/02_EDA.py
python code/03_models_training.py
python code/04_model_comparison.py
python code/05_fairness_audit.py
python code/06_final_model_save.py
```

### 4. Launch Streamlit dashboard

```bash
streamlit run app/app.py
```

Open `http://localhost:8501` in your browser.

---

## Dashboard Pages

| Page | What It Shows |
|------|--------------|
| Dataset Preview | 62-country AI Index table + score distribution by region |
| Country Forecast | Select any country → LSTM forecast chart + RMSE metric |
| Fairness Audit | AEQUITAS metrics table + bar chart + verdict cards (FAIR/BIASED) |

---

## Models Compared

| Model   | Library                | Type                     | Best For                    |
|---------|------------------------|--------------------------|-----------------------------|
| ARIMA   | statsmodels / pmdarima | Statistical              | Non-seasonal short series   |
| SARIMA  | pmdarima               | Seasonal Statistical     | Quarterly seasonal patterns |
| SARIMAX | statsmodels            | Seasonal + Exogenous     | External factor influence   |
| Prophet | Meta Prophet           | Decomposition            | Trend + seasonality         |
| LSTM    | TensorFlow / Keras     | Deep Learning            | Non-linear temporal patterns|

---

## Responsible AI Evaluation

Fairness auditing is implemented using the **AEQUITAS framework** (University of Chicago), which evaluates whether a model makes predictions that are equally fair across different demographic or geographic groups.

### Binary Classification Setup

- **Threshold** = Median Total score of all 62 countries
- **Label 1** = Countries with Total score > threshold (High AI readiness)
- **Label 0** = Countries with Total score ≤ threshold (Low AI readiness)
- **Sensitive attributes audited:** Region (5 groups) + Income group (3 groups)

### Fairness Metrics

| Metric | Meaning |
|--------|---------|
| PPR | Predicted Positive Rate — how often model predicts *high readiness* |
| FPR | False Positive Rate — how often model *wrongly* predicts high readiness |
| FNR | False Negative Rate — how often model *misses* a genuinely high-readiness country |

### Verdict Rule

```
FPR > 0.20  →  BIASED (model over-predicts this group)
PPR = 0.000 →  BIASED (model never predicts this group as positive)
Otherwise   →  FAIR
```

---

## Results

### RMSE Model Comparison
![RMSE Comparison](results/rmse_comparison.png)

### ARIMA Forecast vs Actuals
![Forecast](results/forecast_plot.png)

### AEQUITAS Fairness Audit
![Fairness Audit](results/fairness_audit.png)

---

## Technologies Used

| Category      | Technology         | Purpose                        |
|---------------|--------------------|--------------------------------|
| Language      | Python 3.10+       | Core development               |
| Data          | Pandas + NumPy     | Data manipulation              |
| EDA           | AutoViz            | Automated EDA + FixDQ          |
| Stats Models  | statsmodels        | ARIMA, SARIMA, SARIMAX         |
| Auto ARIMA    | pmdarima           | Automatic order selection      |
| Forecasting   | Prophet            | Trend + seasonality            |
| Deep Learning | TensorFlow / Keras | LSTM model                     |
| Fairness      | AEQUITAS           | Responsible AI audit           |
| Visualization | Matplotlib/Seaborn | Charts + dark theme plots      |
| Dashboard     | Streamlit          | Interactive web application    |

---

## Notes

- Dataset contains **one snapshot per country** — synthetic quarterly series generated from 2015-Q1 to 2024-Q4
- SARIMAX failed due to synthetic random exogenous variable — acknowledged limitation
- Africa FNR = N/A is mathematically correct (no true positives exist to miss)
- LSTM weights saved automatically using Keras `ModelCheckpoint`


*If you found this project useful, consider giving the repository a ⭐ on GitHub*
