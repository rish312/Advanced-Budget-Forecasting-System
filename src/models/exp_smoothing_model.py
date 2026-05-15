"""
Exponential Smoothing (Holt-Winters) Model
============================================
Triple Exponential Smoothing with auto seasonality selection.
"""
import os, sys, warnings
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
warnings.filterwarnings("ignore")


class ExpSmoothingForecaster:
    def __init__(self):
        self.model = None
        self.model_name = "exp_smoothing"
        self.train_data = None
        self.fitted_values = None
        self.best_seasonal = None
        self.aic = None

    def fit(self, train_series: pd.Series) -> "ExpSmoothingForecaster":
        self.train_data = train_series.copy()
        print(f"\n🔧 Fitting Exponential Smoothing model...")
        results = {}
        for seasonal in ["add", "mul"]:
            try:
                m = ExponentialSmoothing(
                    train_series,
                    seasonal_periods=config.EXP_SMOOTHING_CONFIG["seasonal_periods"],
                    trend=config.EXP_SMOOTHING_CONFIG["trend"],
                    seasonal=seasonal,
                    damped_trend=config.EXP_SMOOTHING_CONFIG["damped_trend"],
                    initialization_method="estimated",
                )
                r = m.fit(optimized=True, remove_bias=True)
                results[seasonal] = r
                print(f"   {seasonal.upper()} — AIC: {r.aic:.2f}")
            except Exception as e:
                print(f"   ⚠️ {seasonal} failed: {e}")
        if not results:
            raise RuntimeError("Failed to fit any ETS variant")
        self.best_seasonal = min(results, key=lambda k: results[k].aic)
        self.model = results[self.best_seasonal]
        self.aic = self.model.aic
        self.fitted_values = pd.Series(self.model.fittedvalues, index=train_series.index, name="fitted")
        print(f"   ✅ Selected: {self.best_seasonal.upper()} seasonality (AIC={self.aic:.2f})")
        return self

    def predict(self, horizon: int = None) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("Model not fitted.")
        if horizon is None:
            horizon = config.FORECAST_HORIZON
        forecast = self.model.forecast(horizon)
        last_date = self.train_data.index[-1]
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=horizon, freq=config.DATA_FREQUENCY)
        residual_std = (self.train_data - self.fitted_values).std()
        steps = np.arange(1, horizon + 1)
        width = 1.96 * residual_std * np.sqrt(steps)
        return pd.DataFrame({
            "date": future_dates, "forecast": forecast.values,
            "lower_ci": forecast.values - width, "upper_ci": forecast.values + width,
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
        metrics = {"model": self.model_name, "seasonality": self.best_seasonal,
                    "mae": round(mae, 2), "rmse": round(rmse, 2), "mape": round(mape, 2),
                    "r_squared": round(r2, 4), "aic": round(self.aic, 2)}
        print(f"\n📊 Exp Smoothing Evaluation:")
        for k, v in metrics.items():
            print(f"   {k}: {v}")
        return metrics
