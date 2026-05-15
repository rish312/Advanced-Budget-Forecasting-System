"""
ARIMA / SARIMA Forecasting Model
==================================
Implements Auto-ARIMA with seasonal support for financial time series forecasting.
Uses pmdarima for automatic order selection.
"""

import os
import sys
import pandas as pd
import numpy as np
import warnings

import pmdarima as pm
from statsmodels.tsa.stattools import adfuller

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from src.utils import prepare_time_series

warnings.filterwarnings("ignore")


class ARIMAForecaster:
    """
    ARIMA/SARIMA forecaster with automatic order selection.

    Attributes:
        model: Fitted pmdarima ARIMA model
        model_name: Identifier string
        order: (p, d, q) order
        seasonal_order: (P, D, Q, m) seasonal order
    """

    def __init__(self):
        self.model = None
        self.model_name = "arima"
        self.order = None
        self.seasonal_order = None
        self.train_data = None
        self.fitted_values = None

    def stationarity_test(self, series: pd.Series) -> dict:
        """
        Perform Augmented Dickey-Fuller test for stationarity.

        Args:
            series: Time series to test

        Returns:
            Dict with test statistic, p-value, and conclusion
        """
        result = adfuller(series.dropna(), autolag="AIC")
        return {
            "test_statistic": round(result[0], 4),
            "p_value": round(result[1], 6),
            "lags_used": result[2],
            "observations": result[3],
            "critical_values": {k: round(v, 4) for k, v in result[4].items()},
            "is_stationary": result[1] < 0.05,
        }

    def fit(self, train_series: pd.Series) -> "ARIMAForecaster":
        """
        Fit Auto-ARIMA model to training data.

        Args:
            train_series: Training time series (pd.Series with DatetimeIndex)

        Returns:
            self
        """
        self.train_data = train_series.copy()

        print(f"\n🔧 Fitting ARIMA model...")

        # Stationarity check
        adf = self.stationarity_test(train_series)
        print(f"   ADF Test: stat={adf['test_statistic']}, p={adf['p_value']}")
        print(f"   Stationary: {'Yes ✅' if adf['is_stationary'] else 'No ❌ (differencing needed)'}")

        # Auto-ARIMA
        self.model = pm.auto_arima(
            train_series,
            seasonal=config.ARIMA_CONFIG["seasonal"],
            m=config.ARIMA_CONFIG["m"],
            max_p=config.ARIMA_CONFIG["max_p"],
            max_q=config.ARIMA_CONFIG["max_q"],
            max_d=config.ARIMA_CONFIG["max_d"],
            stepwise=config.ARIMA_CONFIG["stepwise"],
            suppress_warnings=config.ARIMA_CONFIG["suppress_warnings"],
            error_action="ignore",
            trace=False,
        )

        self.order = self.model.order
        self.seasonal_order = self.model.seasonal_order
        self.fitted_values = pd.Series(
            self.model.predict_in_sample(),
            index=train_series.index,
            name="fitted",
        )

        print(f"   ✅ Best order: ARIMA{self.order}")
        print(f"   ✅ Seasonal order: {self.seasonal_order}")
        print(f"   AIC: {self.model.aic():.2f}")

        return self

    def predict(self, horizon: int = None) -> pd.DataFrame:
        """
        Generate forecasts with confidence intervals.

        Args:
            horizon: Number of periods to forecast (default from config)

        Returns:
            DataFrame with columns: date, forecast, lower_ci, upper_ci
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if horizon is None:
            horizon = config.FORECAST_HORIZON

        forecast, conf_int = self.model.predict(
            n_periods=horizon,
            return_conf_int=True,
            alpha=0.05,
        )

        # Generate future dates
        last_date = self.train_data.index[-1]
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=horizon,
            freq=config.DATA_FREQUENCY,
        )

        result = pd.DataFrame({
            "date": future_dates,
            "forecast": forecast,
            "lower_ci": conf_int[:, 0],
            "upper_ci": conf_int[:, 1],
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

        # R-squared
        ss_res = np.sum((actual_values - pred_values) ** 2)
        ss_tot = np.sum((actual_values - np.mean(actual_values)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        metrics = {
            "model": self.model_name,
            "order": str(self.order),
            "seasonal_order": str(self.seasonal_order),
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "mape": round(mape, 2),
            "r_squared": round(r_squared, 4),
            "aic": round(self.model.aic(), 2),
        }

        print(f"\n📊 ARIMA Evaluation:")
        for k, v in metrics.items():
            print(f"   {k}: {v}")

        return metrics

    def get_diagnostics(self) -> dict:
        """Return model diagnostics summary."""
        if self.model is None:
            return {}
        return {
            "order": self.order,
            "seasonal_order": self.seasonal_order,
            "aic": self.model.aic(),
            "bic": self.model.bic(),
            "params": dict(self.model.params()),
        }
