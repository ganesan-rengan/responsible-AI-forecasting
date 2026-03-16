# Responsible AI Forecasting Framework for E-Governance

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Responsible AI](https://img.shields.io/badge/AI-Fairness-green)

End-to-end **Responsible AI pipeline** for forecasting AI readiness indicators using the **AI Global Index dataset**.
The project integrates **time-series forecasting models, model evaluation, fairness auditing, and a deployed Streamlit dashboard**.

---

## Project Overview

This project demonstrates how **Responsible AI principles can be integrated into forecasting systems used in governance and policy decision environments**.

Key components include:

* Time-series forecasting using statistical and deep learning models
* Model performance comparison using MAE and RMSE metrics
* Fairness auditing across global regions using AEQUITAS
* Deployment of the prediction system using a Streamlit dashboard

---

## Dataset

This project uses the **AI Global Index dataset**, which measures how prepared countries are to develop and adopt artificial intelligence.

Dataset Source
https://www.kaggle.com/datasets/katerynameleshenko/ai-index

The dataset includes:

* AI readiness index score
* Country-level indicators
* Regional grouping
* Multiple AI capability factors

---

## Pipeline Architecture

Dataset
↓
Data Preprocessing
↓
Exploratory Data Analysis (EDA)
↓
Forecasting Models

• ARIMA
• SARIMA
• SARIMAX
• Prophet
• LSTM

↓
Model Comparison (RMSE Evaluation)
↓
Best Model Selection
↓
Fairness Audit (AEQUITAS)
↓
Responsible AI Evaluation
↓
Streamlit Deployment Dashboard

---

## Project Structure

```
app/
└── app.py                    # Streamlit dashboard

checkpoint/
└── lstm_checkpoint.h5

code/
├── 01_data_preprocessing.py
├── 02_EDA.py
├── 03_models_training.py
├── 04_model_comparison.py
├── 05_fairness_audit.py
└── 06_final_model_save.py

data/
└── AI_index_db.csv

model/
└── lstm_model.h5

results/
├── rmse_comparison.png
├── forecast_plot.png
└── fairness_audit.png

requirements.txt
README.md
```

---

## Quickstart

Install dependencies

```
pip install -r requirements.txt
```

Place dataset

```
cp /path/to/AI_index_db.csv data/
```

Run the pipeline

```
python code/01_data_preprocessing.py
python code/02_EDA.py
python code/03_models_training.py
python code/04_model_comparison.py
python code/05_fairness_audit.py
python code/06_final_model_save.py
```

Launch Streamlit dashboard

```
streamlit run app/app.py
```

---

## Repository Usage

This repository can be used for:

* Learning time-series forecasting using multiple models
* Understanding Responsible AI evaluation workflows
* Experimenting with fairness auditing in ML systems
* Demonstrating a deployable ML pipeline using Streamlit

---

## Models Compared

| Model   | Library                | Type                     |
| ------- | ---------------------- | ------------------------ |
| ARIMA   | statsmodels / pmdarima | Statistical              |
| SARIMA  | pmdarima               | Seasonal                 |
| SARIMAX | statsmodels            | With exogenous variables |
| Prophet | Meta Prophet           | Decomposition            |
| LSTM    | TensorFlow / Keras     | Deep Learning            |

---

## Responsible AI Evaluation

Fairness auditing in this project is implemented using the **AEQUITAS framework**, which evaluates potential disparities in model predictions across different groups.

The fairness pipeline converts regression predictions into binary outcomes using a median threshold and computes metrics such as:

- Predicted Positive Rate (PPR)
- False Positive Rate (FPR)
- False Negative Rate (FNR)

These metrics are analyzed across groups derived from the dataset (e.g., geographic regions).

**Note:**  
In the current dataset configuration, the processed data resulted in a **single region group during evaluation**. Since fairness auditing requires multiple groups to compute disparity metrics, the analysis produced limited comparative results. However, the full fairness auditing pipeline using AEQUITAS was successfully implemented as part of the Responsible AI framework.

---

## Results

### RMSE Model Comparison

![RMSE Comparison](results/rmse_comparison.png)

### Forecast Visualization

![Forecast](results/forecast_plot.png)

### Fairness Audit

![Fairness](results/fairness_audit.png)

---

## Technologies Used

Python • Pandas • NumPy • Scikit-learn • TensorFlow/Keras • Statsmodels
Prophet • AEQUITAS • Streamlit • AutoViz

---

## Notes

* Dataset contains **one row per country**
* Synthetic quarterly time series generated **2015-Q1 → 2024-Q4**
* LSTM weights saved automatically using **Keras ModelCheckpoint**

---

## Support

If you found this project useful, consider giving the repository a ⭐ on GitHub.
