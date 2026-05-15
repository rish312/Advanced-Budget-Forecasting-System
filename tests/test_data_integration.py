"""Tests for data integration ETL pipeline."""
import os, sys
import pytest
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from data.generate_data import generate_erp_data, generate_crm_data
from src.data_integration import clean_dataframe, merge_data


class TestDataGeneration:
    def test_erp_data_shape(self):
        df = generate_erp_data()
        assert len(df) == config.NUM_MONTHS
        assert "revenue" in df.columns
        assert "net_income" in df.columns

    def test_erp_data_positive_revenue(self):
        df = generate_erp_data()
        assert (df["revenue"] > 0).all()

    def test_crm_data_shape(self):
        erp = generate_erp_data()
        crm = generate_crm_data(erp)
        assert len(crm) == config.NUM_MONTHS
        assert "pipeline_value" in crm.columns

    def test_crm_closed_deals_positive(self):
        erp = generate_erp_data()
        crm = generate_crm_data(erp)
        assert (crm["closed_deals"] > 0).all()


class TestDataCleaning:
    def test_no_missing_values(self):
        df = generate_erp_data()
        cleaned = clean_dataframe(df, "test")
        assert cleaned.isnull().sum().sum() == 0

    def test_sorted_by_date(self):
        df = generate_erp_data()
        cleaned = clean_dataframe(df, "test")
        assert cleaned["date"].is_monotonic_increasing


class TestMerging:
    def test_merge_shape(self):
        erp = generate_erp_data()
        crm = generate_crm_data(erp)
        erp = clean_dataframe(erp, "erp")
        crm = clean_dataframe(crm, "crm")
        merged = merge_data(erp, crm)
        assert len(merged) == config.NUM_MONTHS
        assert "gross_margin_pct" in merged.columns
        assert "net_margin_pct" in merged.columns
