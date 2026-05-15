"""
Budget Forecasting — Full Pipeline Runner
===========================================
Executes the end-to-end pipeline: data generation → ETL → EDA → forecasting
→ variance analysis → save reports for the dashboard.
"""
import os, sys
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data.generate_data import generate_all_data
from src.data_integration import run_etl_pipeline, load_consolidated
from src.eda import run_eda
from src.utils import prepare_time_series, train_test_split_ts, save_dataframe
from src.models.arima_model import ARIMAForecaster
from src.models.prophet_model import ProphetForecaster
from src.models.exp_smoothing_model import ExpSmoothingForecaster
from src.models.ensemble_model import EnsembleForecaster
from src.variance_analysis import run_variance_analysis


def main():
    print("\n" + "🚀" * 30)
    print("   ADVANCED BUDGET FORECASTING — FULL PIPELINE")
    print("🚀" * 30 + "\n")

    # ─── Step 1: Generate Data ───────────────────────────────────────────
    print("\n" + "━" * 60)
    print("STEP 1/6: Generating synthetic financial data")
    print("━" * 60)
    generate_all_data()

    # ─── Step 2: ETL Pipeline ────────────────────────────────────────────
    print("\n" + "━" * 60)
    print("STEP 2/6: Running ETL pipeline")
    print("━" * 60)
    consolidated = run_etl_pipeline()

    # ─── Step 3: EDA ─────────────────────────────────────────────────────
    print("\n" + "━" * 60)
    print("STEP 3/6: Exploratory Data Analysis")
    print("━" * 60)
    run_eda(consolidated)

    # ─── Step 4: Forecasting Models ──────────────────────────────────────
    print("\n" + "━" * 60)
    print("STEP 4/6: Training forecasting models")
    print("━" * 60)

    # Prepare revenue time series
    revenue_ts = prepare_time_series(consolidated, "date", "revenue")
    train, test = train_test_split_ts(revenue_ts)

    print(f"\n📐 Train: {len(train)} months ({train.index[0].strftime('%Y-%m')} → {train.index[-1].strftime('%Y-%m')})")
    print(f"📐 Test:  {len(test)} months ({test.index[0].strftime('%Y-%m')} → {test.index[-1].strftime('%Y-%m')})")

    # Fit individual models and collect metrics
    all_metrics = []
    all_forecasts = []

    # ARIMA
    arima = ARIMAForecaster()
    arima.fit(train)
    arima_metrics = arima.evaluate(test)
    all_metrics.append(arima_metrics)
    arima_fc = arima.predict(len(test))
    all_forecasts.append(arima_fc)

    # Prophet
    prophet = ProphetForecaster()
    prophet.fit(train)
    prophet_metrics = prophet.evaluate(test)
    all_metrics.append(prophet_metrics)
    prophet_fc = prophet.predict(len(test))
    all_forecasts.append(prophet_fc)

    # Exponential Smoothing
    ets = ExpSmoothingForecaster()
    ets.fit(train)
    ets_metrics = ets.evaluate(test)
    all_metrics.append(ets_metrics)
    ets_fc = ets.predict(len(test))
    all_forecasts.append(ets_fc)

    # Ensemble
    ensemble = EnsembleForecaster()
    ensemble.fit(train)
    ensemble.optimize_weights(test)
    ensemble_metrics = ensemble.evaluate(test)
    all_metrics.append(ensemble_metrics)
    ensemble_fc = ensemble.predict(len(test))
    all_forecasts.append(ensemble_fc)

    # Save model comparison
    metrics_df = pd.DataFrame(all_metrics)
    save_dataframe(metrics_df, "model_comparison", formats=["csv", "json"])

    # Combine all forecasts
    forecasts_combined = pd.concat(all_forecasts, ignore_index=True)
    save_dataframe(forecasts_combined, "all_forecasts", formats=["csv"])

    # Print comparison table
    print("\n" + "=" * 60)
    print("  MODEL COMPARISON")
    print("=" * 60)
    print(metrics_df[["model", "mae", "rmse", "mape", "r_squared"]].to_string(index=False))

    # ─── Step 5: Variance Analysis ───────────────────────────────────────
    print("\n" + "━" * 60)
    print("STEP 5/6: Variance Analysis")
    print("━" * 60)

    # Use ensemble forecast for variance analysis
    ensemble_preds = ensemble.predict(horizon=len(test))
    forecast_series = pd.Series(
        ensemble_preds["forecast"].values,
        index=test.index,
        name="forecast",
    )

    variance_df, variance_summary = run_variance_analysis(
        test, forecast_series, metric_name="revenue", model_name="ensemble"
    )

    # ─── Step 6: Summary ─────────────────────────────────────────────────
    print("\n" + "━" * 60)
    print("STEP 6/6: Pipeline Complete!")
    print("━" * 60)

    best_model = metrics_df.loc[metrics_df["mape"].idxmin()]
    print(f"\n🏆 Best Model: {best_model['model'].upper()} (MAPE: {best_model['mape']:.2f}%)")
    print(f"\n📊 Dashboard ready! Start with:")
    print(f"   python app/app.py")
    print(f"\n   Then open: http://localhost:{config.FLASK_PORT}")

    return metrics_df, variance_df, variance_summary


if __name__ == "__main__":
    main()
