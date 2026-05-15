"""Tests for variance analysis module."""
import os, sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.variance_analysis import compute_variance, decompose_variance, generate_variance_summary


@pytest.fixture
def sample_data():
    dates = pd.date_range("2024-01-01", periods=6, freq="MS")
    actuals = pd.Series([100, 110, 105, 120, 115, 130], index=dates, name="revenue")
    forecasts = pd.Series([95, 108, 110, 115, 120, 125], index=dates, name="forecast")
    return actuals, forecasts


class TestVarianceComputation:
    def test_variance_values(self, sample_data):
        actuals, forecasts = sample_data
        result = compute_variance(actuals, forecasts)
        assert len(result) == 6
        assert "variance" in result.columns
        # First row: 100 - 95 = 5
        assert result.iloc[0]["variance"] == pytest.approx(5.0)

    def test_classification_revenue(self, sample_data):
        actuals, forecasts = sample_data
        result = compute_variance(actuals, forecasts, metric_name="revenue")
        # Positive variance for revenue = Favorable
        fav = result[result["variance"] > 0]
        assert (fav["classification"] == config.VARIANCE_FAVORABLE_LABEL).all()

    def test_classification_cost(self, sample_data):
        actuals, forecasts = sample_data
        result = compute_variance(actuals, forecasts, metric_name="opex")
        # Positive variance for cost = Adverse
        adv = result[result["variance"] > 0]
        assert (adv["classification"] == config.VARIANCE_ADVERSE_LABEL).all()


class TestDecomposition:
    def test_drivers_present(self, sample_data):
        actuals, forecasts = sample_data
        var_df = compute_variance(actuals, forecasts)
        decomposed = decompose_variance(var_df)
        assert "volume_driver" in decomposed.columns
        assert "price_mix_driver" in decomposed.columns
        assert "timing_driver" in decomposed.columns


class TestSummary:
    def test_summary_keys(self, sample_data):
        actuals, forecasts = sample_data
        var_df = compute_variance(actuals, forecasts)
        summary = generate_variance_summary(var_df)
        assert "total_periods" in summary
        assert "favorable_count" in summary
        assert "needs_retrain" in summary
        assert summary["total_periods"] == 6
