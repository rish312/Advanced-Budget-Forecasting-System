"""
Ensemble Forecasting Model
============================
Weighted ensemble combining ARIMA, Prophet, and Exponential Smoothing forecasts.
Weights are optimized by inverse MAPE on validation data.
"""
import os, sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from src.models.arima_model import ARIMAForecaster
from src.models.prophet_model import ProphetForecaster
from src.models.exp_smoothing_model import ExpSmoothingForecaster


class EnsembleForecaster:
    def __init__(self):
        self.model_name = "ensemble"
        self.models = {}
        self.weights = config.ENSEMBLE_CONFIG["default_weights"].copy()
        self.train_data = None
        self.individual_forecasts = {}
        self.individual_metrics = {}

    def fit(self, train_series: pd.Series) -> "EnsembleForecaster":
        self.train_data = train_series.copy()
        print("\n" + "=" * 60)
        print("  ENSEMBLE MODEL — FITTING ALL COMPONENTS")
        print("=" * 60)
        # Fit ARIMA
        arima = ARIMAForecaster()
        arima.fit(train_series)
        self.models["arima"] = arima
        # Fit Prophet
        prophet = ProphetForecaster()
        prophet.fit(train_series)
        self.models["prophet"] = prophet
        # Fit Exp Smoothing
        ets = ExpSmoothingForecaster()
        ets.fit(train_series)
        self.models["exp_smoothing"] = ets
        print(f"\n✅ All 3 component models fitted")
        return self

    def optimize_weights(self, validation_series: pd.Series):
        """Optimize weights using inverse MAPE on validation data."""
        print("\n⚖️ Optimizing ensemble weights...")
        mapes = {}
        for name, model in self.models.items():
            metrics = model.evaluate(validation_series)
            mapes[name] = max(metrics["mape"], 0.01)  # avoid zero division
            self.individual_metrics[name] = metrics
        inv_mapes = {k: 1.0 / v for k, v in mapes.items()}
        total = sum(inv_mapes.values())
        self.weights = {k: round(v / total, 4) for k, v in inv_mapes.items()}
        print(f"   Optimized weights: {self.weights}")
        return self.weights

    def predict(self, horizon: int = None) -> pd.DataFrame:
        if not self.models:
            raise ValueError("Models not fitted.")
        if horizon is None:
            horizon = config.FORECAST_HORIZON
        forecasts = {}
        for name, model in self.models.items():
            pred = model.predict(horizon=horizon)
            forecasts[name] = pred
            self.individual_forecasts[name] = pred
        # Weighted average
        dates = forecasts["arima"]["date"]
        ensemble_forecast = np.zeros(horizon)
        ensemble_lower = np.zeros(horizon)
        ensemble_upper = np.zeros(horizon)
        for name, pred in forecasts.items():
            w = self.weights.get(name, 1.0 / len(forecasts))
            ensemble_forecast += w * pred["forecast"].values
            ensemble_lower += w * pred["lower_ci"].values
            ensemble_upper += w * pred["upper_ci"].values
        return pd.DataFrame({
            "date": dates, "forecast": ensemble_forecast,
            "lower_ci": ensemble_lower, "upper_ci": ensemble_upper,
            "model": self.model_name,
        })

    def evaluate(self, test_series: pd.Series) -> dict:
        preds = self.predict(horizon=len(test_series))
        pv, av = preds["forecast"].values, test_series.values
        mae = np.mean(np.abs(av - pv))
        rmse = np.sqrt(np.mean((av - pv) ** 2))
        mape = np.mean(np.abs((av - pv) / av)) * 100
        ss_res = np.sum((av - pv) ** 2)
        ss_tot = np.sum((av - np.mean(av)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        metrics = {"model": self.model_name, "weights": str(self.weights),
                    "mae": round(mae, 2), "rmse": round(rmse, 2),
                    "mape": round(mape, 2), "r_squared": round(r2, 4)}
        print(f"\n📊 Ensemble Evaluation:")
        for k, v in metrics.items():
            print(f"   {k}: {v}")
        return metrics

    def get_comparison_table(self) -> pd.DataFrame:
        """Get a comparison table of all models + ensemble."""
        rows = list(self.individual_metrics.values())
        ensemble_metrics = {"model": "ensemble", "weights": str(self.weights)}
        rows.append(ensemble_metrics)
        return pd.DataFrame(rows)
