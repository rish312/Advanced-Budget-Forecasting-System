"""Tests for forecasting models."""
import os, sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from data.generate_data import generate_erp_data
from src.utils import prepare_time_series, train_test_split_ts
from src.models.arima_model import ARIMAForecaster
from src.models.exp_smoothing_model import ExpSmoothingForecaster


@pytest.fixture
def revenue_series():
    df = generate_erp_data()
    df["date"] = pd.to_datetime(df["date"])
    ts = df.set_index("date")["revenue"]
    ts.index = pd.DatetimeIndex(ts.index, freq=config.DATA_FREQUENCY)
    return ts


@pytest.fixture
def train_test(revenue_series):
    return train_test_split_ts(revenue_series)


class TestARIMA:
    def test_fit_returns_self(self, train_test):
        train, _ = train_test
        model = ARIMAForecaster()
        result = model.fit(train)
        assert result is model
        assert model.model is not None

    def test_predict_shape(self, train_test):
        train, _ = train_test
        model = ARIMAForecaster()
        model.fit(train)
        preds = model.predict(horizon=6)
        assert len(preds) == 6
        assert "forecast" in preds.columns
        assert "lower_ci" in preds.columns

    def test_evaluate_metrics(self, train_test):
        train, test = train_test
        model = ARIMAForecaster()
        model.fit(train)
        metrics = model.evaluate(test)
        assert "mae" in metrics
        assert "mape" in metrics
        assert metrics["mape"] >= 0


class TestExpSmoothing:
    def test_fit_returns_self(self, train_test):
        train, _ = train_test
        model = ExpSmoothingForecaster()
        result = model.fit(train)
        assert result is model

    def test_predict_shape(self, train_test):
        train, _ = train_test
        model = ExpSmoothingForecaster()
        model.fit(train)
        preds = model.predict(horizon=6)
        assert len(preds) == 6
