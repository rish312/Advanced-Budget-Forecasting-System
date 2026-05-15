"""
Prophet Forecasting Model
==========================
Implements Facebook Prophet for financial time series forecasting
with support for seasonality, holidays, and trend changepoints.
"""

import os
import sys
import pandas as pd
import numpy as np
import warnings
import logging

from prophet import Prophet

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

# Suppress Prophet's verbose logging
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
warnings.filterwarnings("ignore")


def _get_fiscal_events() -> pd.DataFrame:
    """Generate fiscal calendar events (quarter ends, year end)."""
    events = []
    for year in range(2019, 2027):
        events.extend([
            {"holiday": "Q1_Close", "ds": f"{year}-03-31", "lower_window": -5, "upper_window": 0},
            {"holiday": "Q2_Close", "ds": f"{year}-06-30", "lower_window": -5, "upper_window": 0},
            {"holiday": "Q3_Close", "ds": f"{year}-09-30", "lower_window": -5, "upper_window": 0},
            {"holiday": "Year_End", "ds": f"{year}-12-31", "lower_window": -10, "upper_window": 0},
        ])
    return pd.DataFrame(events)


class ProphetForecaster:
    """
    Facebook Prophet forecaster for financial data.

    Handles yearly/quarterly seasonality, fiscal events,
    and automatic trend changepoint detection.

    Attributes:
        model: Fitted Prophet model
        model_name: Identifier string
    """

    def __init__(self):
        self.model = None
        self.model_name = "prophet"
        self.train_data = None
        self.fitted_values = None

    def fit(self, train_series: pd.Series) -> "ProphetForecaster":
        """
        Fit Prophet model to training data.

        Args:
            train_series: Training time series (pd.Series with DatetimeIndex)

        Returns:
            self
        """
        self.train_data = train_series.copy()

        print(f"\n🔧 Fitting Prophet model...")

        # Prepare data in Prophet format
        prophet_df = pd.DataFrame({
            "ds": train_series.index,
            "y": train_series.values,
        })

        # Initialize Prophet with config
        self.model = Prophet(
            yearly_seasonality=config.PROPHET_CONFIG["yearly_seasonality"],
            weekly_seasonality=config.PROPHET_CONFIG["weekly_seasonality"],
            daily_seasonality=config.PROPHET_CONFIG["daily_seasonality"],
            seasonality_mode=config.PROPHET_CONFIG["seasonality_mode"],
            changepoint_prior_scale=config.PROPHET_CONFIG["changepoint_prior_scale"],
            interval_width=config.PROPHET_CONFIG["interval_width"],
            holidays=_get_fiscal_events(),
        )

        # Add quarterly seasonality
        self.model.add_seasonality(
            name="quarterly",
            period=91.25,
            fourier_order=5,
        )

        # Fit
        self.model.fit(prophet_df)

        # Get in-sample predictions
        in_sample = self.model.predict(prophet_df)
        self.fitted_values = pd.Series(
            in_sample["yhat"].values,
            index=train_series.index,
            name="fitted",
        )

        changepoints = self.model.changepoints
        print(f"   ✅ Prophet model fitted")
        print(f"   Changepoints detected: {len(changepoints)}")
        print(f"   Seasonality mode: {config.PROPHET_CONFIG['seasonality_mode']}")

        return self

    def predict(self, horizon: int = None) -> pd.DataFrame:
        """
        Generate forecasts with uncertainty intervals.

        Args:
            horizon: Number of periods to forecast (default from config)

        Returns:
            DataFrame with columns: date, forecast, lower_ci, upper_ci
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if horizon is None:
            horizon = config.FORECAST_HORIZON

        # Create future dataframe
        future = self.model.make_future_dataframe(
            periods=horizon,
            freq=config.DATA_FREQUENCY,
        )

        forecast = self.model.predict(future)

        # Extract only the forecast period
        forecast_period = forecast.tail(horizon)

        result = pd.DataFrame({
            "date": forecast_period["ds"].values,
            "forecast": forecast_period["yhat"].values,
            "lower_ci": forecast_period["yhat_lower"].values,
            "upper_ci": forecast_period["yhat_upper"].values,
            "model": self.model_name,
        })

        return result

    def evaluate(self, test_series: pd.Series) -> dict:
        """
        Evaluate model against test data.

        Args:
            test_series: Actual values for the forecast period

        Returns:
            Dict of evaluation metrics
        """
        predictions = self.predict(horizon=len(test_series))
        pred_values = predictions["forecast"].values
        actual_values = test_series.values

        mae = np.mean(np.abs(actual_values - pred_values))
        rmse = np.sqrt(np.mean((actual_values - pred_values) ** 2))
        mape = np.mean(np.abs((actual_values - pred_values) / actual_values)) * 100

        ss_res = np.sum((actual_values - pred_values) ** 2)
        ss_tot = np.sum((actual_values - np.mean(actual_values)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        metrics = {
            "model": self.model_name,
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "mape": round(mape, 2),
            "r_squared": round(r_squared, 4),
        }

        print(f"\n📊 Prophet Evaluation:")
        for k, v in metrics.items():
            print(f"   {k}: {v}")

        return metrics

    def get_components(self) -> dict:
        """Return decomposed forecast components."""
        if self.model is None:
            return {}

        prophet_df = pd.DataFrame({
            "ds": self.train_data.index,
            "y": self.train_data.values,
        })
        forecast = self.model.predict(prophet_df)

        return {
            "trend": forecast["trend"].values,
            "yearly": forecast.get("yearly", pd.Series([0])).values,
            "quarterly": forecast.get("quarterly", pd.Series([0])).values,
        }
