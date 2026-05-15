"""
Budget Forecasting — Flask Web Dashboard
==========================================
Interactive dashboard with KPI cards, forecast charts, and variance analysis.
"""
import os, sys, json
import pandas as pd
import numpy as np
from flask import Flask, render_template, jsonify, request, Response

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.data_integration import load_consolidated
from src.utils import get_db_connection, prepare_time_series, train_test_split_ts

app = Flask(__name__)

# ─── Data Loading ────────────────────────────────────────────────────────────────
def get_financial_data():
    """Load consolidated financial data."""
    try:
        return load_consolidated()
    except FileNotFoundError:
        return pd.DataFrame()


def get_forecast_data():
    """Load forecast results if available."""
    path = os.path.join(config.REPORTS_DIR, "all_forecasts.csv")
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=["date"])
    return pd.DataFrame()


def get_variance_data():
    """Load variance analysis results if available."""
    path = os.path.join(config.REPORTS_DIR, "variance_revenue.csv")
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=["date"])
    return pd.DataFrame()


def get_metrics_data():
    """Load model comparison metrics."""
    path = os.path.join(config.REPORTS_DIR, "model_comparison.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


# ─── Routes ──────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    df = get_financial_data()
    if df.empty:
        return render_template("index.html", has_data=False)
    # KPI calculations
    latest = df.iloc[-1]
    prev_year = df[df["date"] < (pd.Timestamp(latest["date"]) - pd.DateOffset(years=1))]
    prev_year_latest = prev_year.iloc[-1] if not prev_year.empty else latest
    kpis = {
        "revenue": {"value": latest["revenue"], "change": ((latest["revenue"] - prev_year_latest["revenue"]) / prev_year_latest["revenue"] * 100)},
        "net_income": {"value": latest["net_income"], "change": ((latest["net_income"] - prev_year_latest["net_income"]) / prev_year_latest["net_income"] * 100)},
        "gross_margin": {"value": latest.get("gross_margin_pct", 0), "change": 0},
        "total_expenses": {"value": latest["total_expenses"], "change": ((latest["total_expenses"] - prev_year_latest["total_expenses"]) / prev_year_latest["total_expenses"] * 100)},
    }
    return render_template("index.html", has_data=True, kpis=kpis, latest_date=str(latest["date"])[:10])


@app.route("/forecast")
def forecast():
    return render_template("forecast.html")


@app.route("/variance")
def variance():
    return render_template("variance.html")


# ─── API Endpoints ───────────────────────────────────────────────────────────────
@app.route("/api/financial-data")
def api_financial_data():
    df = get_financial_data()
    if df.empty:
        return jsonify({"error": "No data available"})
    df["date"] = df["date"].astype(str)
    cols = ["date", "revenue", "cogs", "opex", "payroll", "utilities",
            "depreciation", "total_expenses", "net_income"]
    available = [c for c in cols if c in df.columns]
    return jsonify(df[available].to_dict(orient="records"))


@app.route("/api/forecast-data")
def api_forecast_data():
    df = get_forecast_data()
    if df.empty:
        return jsonify({"error": "No forecast data"})
    df["date"] = df["date"].astype(str)
    return Response(df.to_json(orient="records"), mimetype="application/json")


@app.route("/api/variance-data")
def api_variance_data():
    df = get_variance_data()
    if df.empty:
        return jsonify({"error": "No variance data"})
    df["date"] = df["date"].astype(str)
    return jsonify(df.to_dict(orient="records"))


@app.route("/api/metrics-data")
def api_metrics_data():
    df = get_metrics_data()
    if df.empty:
        return jsonify({"error": "No metrics data"})
    df = df.fillna("")
    return Response(df.to_json(orient="records"), mimetype="application/json")


@app.route("/api/kpi-summary")
def api_kpi_summary():
    df = get_financial_data()
    if df.empty:
        return jsonify({"error": "No data"})
    annual = df.groupby("year").agg({
        "revenue": "sum", "net_income": "sum", "total_expenses": "sum",
        "closed_deals": "sum", "cac": "mean",
    }).reset_index()
    annual.columns = ["year", "revenue", "net_income", "total_expenses", "closed_deals", "avg_cac"]
    return jsonify(annual.to_dict(orient="records"))


if __name__ == "__main__":
    app.run(host=config.FLASK_HOST, port=config.FLASK_PORT, debug=config.FLASK_DEBUG)
