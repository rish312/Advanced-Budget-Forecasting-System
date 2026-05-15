"""
Shared Utility Functions
========================
Helper functions used across the forecasting system.
"""

import os
import sqlite3
import pandas as pd
import numpy as np
from typing import Optional

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def get_db_connection() -> sqlite3.Connection:
    """Get a connection to the SQLite database."""
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    """Initialize the database using the SQL schema."""
    schema_path = os.path.join(config.SQL_DIR, "schema.sql")
    conn = get_db_connection()
    with open(schema_path, "r") as f:
        conn.executescript(f.read())
    conn.close()
    print(f"✅ Database initialized at {config.DB_PATH}")


def prepare_time_series(
    df: pd.DataFrame,
    date_col: str = "date",
    value_col: str = "revenue",
) -> pd.Series:
    """
    Convert a DataFrame column into a proper time series.

    Args:
        df: Input DataFrame
        date_col: Name of the date column
        value_col: Name of the value column to extract

    Returns:
        pd.Series with DatetimeIndex
    """
    ts = df.set_index(date_col)[value_col].copy()
    ts.index = pd.DatetimeIndex(ts.index, freq=config.DATA_FREQUENCY)
    ts.name = value_col
    return ts


def train_test_split_ts(
    series: pd.Series,
    split_ratio: float = None,
) -> tuple[pd.Series, pd.Series]:
    """
    Split a time series into train and test sets (temporal split).

    Args:
        series: Time series data
        split_ratio: Fraction for training (default from config)

    Returns:
        Tuple of (train, test) series
    """
    if split_ratio is None:
        split_ratio = config.TRAIN_TEST_SPLIT_RATIO

    split_idx = int(len(series) * split_ratio)
    train = series.iloc[:split_idx]
    test = series.iloc[split_idx:]
    return train, test


def format_currency(value: float) -> str:
    """Format a number as USD currency string."""
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:,.1f}M"
    elif abs(value) >= 1_000:
        return f"${value / 1_000:,.1f}K"
    else:
        return f"${value:,.2f}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """Format a number as a percentage string."""
    return f"{value:.{decimals}f}%"


def ensure_directory(path: str):
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def save_dataframe(
    df: pd.DataFrame,
    filename: str,
    directory: str = None,
    formats: list[str] = None,
):
    """
    Save a DataFrame in one or more formats.

    Args:
        df: DataFrame to save
        filename: Base filename (without extension)
        directory: Target directory (default: REPORTS_DIR)
        formats: List of formats ['csv', 'json', 'excel']
    """
    if directory is None:
        directory = config.REPORTS_DIR
    if formats is None:
        formats = ["csv"]

    ensure_directory(directory)

    for fmt in formats:
        path = os.path.join(directory, f"{filename}.{fmt}")
        if fmt == "csv":
            df.to_csv(path, index=False)
        elif fmt == "json":
            df.to_json(path, orient="records", date_format="iso", indent=2)
        elif fmt in ("excel", "xlsx"):
            path = os.path.join(directory, f"{filename}.xlsx")
            df.to_excel(path, index=False, engine="openpyxl")
        print(f"   📄 Saved {path}")
