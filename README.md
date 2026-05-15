# 📊 Advanced Budget Forecasting System

> Data-driven financial forecasting using historical ERP/CRM data, time series models, and automated variance analysis with an interactive web dashboard.

---

## 🎯 Project Overview

This project implements a modular **budget forecasting pipeline** that consolidates historical financial data, applies multiple predictive models, performs automated variance analysis, and presents results through a premium interactive dashboard.

### Key Capabilities

| Feature | Description |
|---------|-------------|
| **Data Integration** | ETL pipeline consolidating ERP & CRM data into a single source of truth |
| **Multi-Model Forecasting** | ARIMA/SARIMA, Prophet, Exponential Smoothing, and Weighted Ensemble |
| **Variance Analysis** | Automated actuals-vs-forecast comparison with driver decomposition |
| **Feedback Loop** | Auto-retrain trigger when cumulative MAPE exceeds threshold |
| **Interactive Dashboard** | Premium Flask web app with Plotly.js visualizations |

---

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌───────────────────┐
│  Data Sources    │     │   ETL        │     │   Source of Truth  │
│  (ERP + CRM)    │────▶│  Pipeline    │────▶│  (SQLite + CSV)    │
└─────────────────┘     └──────────────┘     └────────┬──────────┘
                                                       │
                              ┌─────────────────────────┤
                              ▼                         ▼
                    ┌──────────────────┐     ┌────────────────────┐
                    │      EDA         │     │  Forecasting       │
                    │  (7 chart types) │     │  Models            │
                    └──────────────────┘     │  ├─ ARIMA/SARIMA   │
                                            │  ├─ Prophet         │
                                            │  ├─ Exp. Smoothing  │
                                            │  └─ Ensemble        │
                                            └────────┬───────────┘
                                                     │
                                                     ▼
                                            ┌────────────────────┐
                                            │  Variance Analysis │
                                            │  (Feedback Loop)   │
                                            └────────┬───────────┘
                                                     │
                                                     ▼
                                            ┌────────────────────┐
                                            │  Web Dashboard     │
                                            │  (Flask + Plotly)  │
                                            └────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd budget_forecasting

# Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Run the Full Pipeline

```bash
python run_pipeline.py
```

This executes all 6 steps:
1. 📊 Generate synthetic financial data (60 months)
2. 🔗 Run ETL pipeline (clean, merge, persist)
3. 📈 Exploratory Data Analysis (7 chart types)
4. 🤖 Train 4 forecasting models + evaluate
5. 📉 Variance analysis with driver decomposition
6. ✅ Generate reports for the dashboard

### Launch the Dashboard

```bash
python app/app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

---

## 📁 Project Structure

```
budget_forecasting/
├── config.py                  # Global configuration
├── run_pipeline.py            # End-to-end pipeline runner
├── requirements.txt           # Dependencies
│
├── data/
│   ├── generate_data.py       # Synthetic data generator
│   ├── raw/                   # Raw ERP & CRM CSVs
│   └── processed/             # Consolidated dataset
│
├── src/
│   ├── data_integration.py    # ETL pipeline
│   ├── eda.py                 # Exploratory analysis
│   ├── evaluation.py          # Model metrics
│   ├── variance_analysis.py   # Actuals vs forecast
│   ├── utils.py               # Shared helpers
│   └── models/
│       ├── arima_model.py     # ARIMA/SARIMA
│       ├── prophet_model.py   # Facebook Prophet
│       ├── exp_smoothing_model.py  # Holt-Winters
│       └── ensemble_model.py  # Weighted ensemble
│
├── app/                       # Flask web dashboard
│   ├── app.py
│   ├── templates/
│   └── static/
│
├── sql/schema.sql             # Database schema
├── outputs/                   # Generated plots & reports
└── tests/                     # Unit tests
```

---

## 🤖 Forecasting Models

### 1. ARIMA/SARIMA
- Automatic order selection via `pmdarima`
- Seasonal component with period=12
- ADF stationarity testing

### 2. Facebook Prophet
- Yearly + quarterly seasonality
- Fiscal calendar events (quarter/year ends)
- Trend changepoint detection

### 3. Exponential Smoothing (Holt-Winters)
- Auto-selects additive vs. multiplicative seasonality
- Damped trend for stability
- AIC-based model selection

### 4. Weighted Ensemble
- Combines all three models
- Weights optimized by inverse MAPE
- Typically achieves lowest forecast error

---

## 📉 Variance Analysis

The system implements a **feedback loop** for continuous forecast improvement:

1. **Compute**: `Variance = Actual − Forecast` per period
2. **Classify**: Favorable (revenue above forecast) vs. Adverse
3. **Decompose**: Volume, Price/Mix, and Timing drivers
4. **Monitor**: Track cumulative MAPE against threshold
5. **Retrain**: Trigger model retraining when accuracy degrades

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test suite
pytest tests/test_models.py -v
pytest tests/test_variance.py -v
```

---

## 🛠️ Tools & Technologies

| Category | Tools |
|----------|-------|
| **Language** | Python 3.10+ |
| **Data** | Pandas, NumPy, SQLite |
| **Visualization** | Matplotlib, Seaborn, Plotly.js |
| **Forecasting** | statsmodels, pmdarima, Prophet |
| **Web** | Flask, HTML/CSS/JavaScript |
| **Testing** | pytest |

---

## 📄 License

This project is for educational and demonstration purposes.
