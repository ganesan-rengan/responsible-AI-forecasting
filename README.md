# Responsible AI Project – AI Index Forecaster

End-to-end time-series forecasting and fairness audit on the
[AI Readiness Index](https://www.tortoisemedia.com/intelligence/global-ai/) dataset.

---

## Project Structure

```
responsible_ai_project/
├── app/
│   └── app.py                    # Streamlit dashboard
├── checkpoint/
│   └── lstm_checkpoint.h5        # Best LSTM weights (saved during training)
├── code/
│   ├── 01_data_preprocessing.py  # Load, clean & date-index the dataset
│   ├── 02_EDA.py                 # AutoViz, ADF test, ACF/PACF plots
│   ├── 03_models_training.py     # ARIMA, SARIMA, SARIMAX, Prophet, LSTM
│   ├── 04_model_comparison.py    # Per-country evaluation & RMSE comparison
│   ├── 05_fairness_audit.py      # Aequitas fairness audit by world region
│   └── 06_final_model_save.py    # Train final LSTM & save model weights
├── data/
│   └── AI_index_db.csv           # Raw dataset (place here before running)
├── model/
│   └── lstm_model.h5             # Final saved LSTM model
├── README.md
└── requirements.txt
```

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place the dataset
cp /path/to/AI_index_db.csv data/

# 3. Run pipeline in order
python code/01_data_preprocessing.py
python code/02_EDA.py
python code/03_models_training.py
python code/04_model_comparison.py
python code/05_fairness_audit.py
python code/06_final_model_save.py

# 4. Launch dashboard
streamlit run app/app.py
```

---

## Models Compared

| Model   | Library       | Type            |
|---------|---------------|-----------------|
| ARIMA   | statsmodels / pmdarima | Univariate |
| SARIMA  | pmdarima      | Seasonal        |
| SARIMAX | statsmodels   | With exogenous  |
| Prophet | Meta Prophet  | Decomposition   |
| LSTM    | TensorFlow/Keras | Deep Learning |

---

## Fairness

Fairness audit uses **Aequitas** to check for disparities in model performance
across world regions (Americas, Europe, Asia, Africa, Oceania).

---

## Notes

- `AI_index_db.csv` has one row per country. Synthetic quarterly time series
  (2015-Q1 → 2024-Q4) are generated per country for model training.
- The `checkpoint/lstm_checkpoint.h5` is created automatically by
  `06_final_model_save.py` via a Keras `ModelCheckpoint` callback.
