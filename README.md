# 📊 Surakarta Tuberculosis (TB) Cases Forecasting: Hybrid ARIMA-ANN Approach

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Domain: Healthcare Analytics](https://img.shields.io/badge/Domain-Healthcare%20Analytics-green.svg)](#)

A comprehensive end-to-end time series analysis and forecasting project aimed at predicting monthly Tuberculosis (TB) incidence in Surakarta, Central Java. This repository demonstrates how a **Hybrid ARIMA-Artificial Neural Network (ANN)** model captures both linear dynamic patterns and non-linear residual dynamics, outperforming standalone time series models.

---

## 📌 Project Overview & Problem Statement

Tuberculosis remains a critical public health priority in urban areas. Accurate incidence forecasting is vital for local health departments (Dinas Kesehatan) to allocate medical resources, manage pharmaceutical supplies, and evaluate intervention programs effectively.

Standard univariate models like **ARIMA** excel at modeling linear stationary autocorrelation patterns. However, real-world epidemic trends often present non-linear fluctuations caused by seasonal outbreaks, social dynamics, and policy changes. By leveraging a **Hybrid ARIMA-ANN architecture**, this project models:
1. **Linear Component ($L_t$):** Modeled using Box-Jenkins ARIMA.
2. **Non-linear Residual Component ($N_t$):** Modeled using a Multi-Layer Perceptron (MLP) Neural Network trained on the ARIMA residuals.

$$\hat{Y}_t = \hat{L}_t + \hat{N}_t$$

---

## 💡 Key Findings & Benchmark Performance

The models were trained on historical monthly case data and evaluated using **Mean Absolute Percentage Error (MAPE)** and **Root Mean Squared Error (RMSE)** on a held-out test dataset:

| Model Architecture | Evaluation MAPE (%) | Evaluation RMSE | Key Advantage |
| :--- | :---: | :---: | :--- |
| **Naive Baseline** | 18.42% | 14.85 | Simple benchmark based on previous period |
| **ARIMA (2,1,1)** | 10.53% | 8.62 | Captures linear autoregressive trends |
| **Hybrid ARIMA-ANN** ⭐ | **6.43%** | **5.18** | **Captures linear trends + non-linear residual signals** |

> **Impact:** The Hybrid ARIMA-ANN model reduced forecasting error by over **38.9%** compared to the classical ARIMA model, achieving high-accuracy predictions (<7% MAPE) suitable for operational healthcare planning.

---

## 📁 Repository Structure

```text
surakarta-tb-forecasting-hybrid/
│
├── data/
│   ├── raw/                 # Original monthly TB cases dataset (CSV)
│   └── processed/           # Cleaned and feature-engineered dataset
│
├── notebooks/
│   └── 01_eda_and_modeling.ipynb # Complete step-by-step Jupyter Notebook
│
├── src/
│   ├── __init__.py
│   ├── data_processing.py   # Data ingestion & stationarity checks (ADF test)
│   ├── models.py            # ARIMA baseline & ANN residual modeling pipeline
│   └── utils.py             # Evaluation metrics (MAPE, RMSE) & visualizers
│
├── reports/
│   └── figures/             # Saved visual assets (ACF/PACF, forecast comparison)
│
├── main.py                  # CLI pipeline runner
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
