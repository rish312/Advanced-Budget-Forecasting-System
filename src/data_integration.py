"""
Data Integration (ETL) Module
================================
Handles loading, cleaning, merging, and persisting of raw financial data
from ERP and CRM sources into a consolidated "source of truth."
"""

import os
import sys
import sqlite3
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.utils import get_db_connection, init_database


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load raw ERP and CRM CSV files.

    Returns:
        Tuple of (erp_df, crm_df)
    """
    erp_path = os.path.join(config.RAW_DATA_DIR, "erp_financials.csv")
    crm_path = os.path.join(config.RAW_DATA_DIR, "crm_metrics.csv")

    erp_df = pd.read_csv(erp_path, parse_dates=["date"])
    crm_df = pd.read_csv(crm_path, parse_dates=["date"])

    print(f"📂 Loaded ERP data: {erp_df.shape[0]} rows, {erp_df.shape[1]} columns")
    print(f"📂 Loaded CRM data: {crm_df.shape[0]} rows, {crm_df.shape[1]} columns")

    return erp_df, crm_df


def clean_dataframe(df: pd.DataFrame, name: str = "data") -> pd.DataFrame:
    """
    Clean a DataFrame: handle missing values, duplicates, and data types.

    Args:
        df: Raw DataFrame
        name: Dataset name for logging

    Returns:
        Cleaned DataFrame
    """
    initial_rows = len(df)

    # Remove duplicates based on date
    df = df.drop_duplicates(subset=["date"], keep="last")
    dupes_removed = initial_rows - len(df)

    # Handle missing values — forward fill then backward fill
    missing_before = df.isnull().sum().sum()
    df = df.ffill().bfill()
    missing_after = df.isnull().sum().sum()

    # Ensure date column is datetime
    df["date"] = pd.to_datetime(df["date"])

    # Sort by date
    df = df.sort_values("date").reset_index(drop=True)

    # Ensure numeric columns are properly typed
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"🧹 Cleaned {name}:")
    print(f"   Duplicates removed: {dupes_removed}")
    print(f"   Missing values filled: {missing_before - missing_after}")
    print(f"   Final shape: {df.shape}")

    return df


def merge_data(
    erp_df: pd.DataFrame,
    crm_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge ERP and CRM data on date to create consolidated dataset.

    Args:
        erp_df: Cleaned ERP DataFrame
        crm_df: Cleaned CRM DataFrame

    Returns:
        Consolidated DataFrame
    """
    consolidated = pd.merge(erp_df, crm_df, on="date", how="left")

    # Add derived metrics
    consolidated["gross_profit"] = (
        consolidated["revenue"] - consolidated["cogs"]
    )
    consolidated["gross_margin_pct"] = (
        consolidated["gross_profit"] / consolidated["revenue"] * 100
    ).round(2)
    consolidated["net_margin_pct"] = (
        consolidated["net_income"] / consolidated["revenue"] * 100
    ).round(2)
    consolidated["opex_ratio"] = (
        consolidated["opex"] / consolidated["revenue"] * 100
    ).round(2)
    consolidated["revenue_per_deal"] = (
        consolidated["revenue"] / consolidated["closed_deals"]
    ).round(2)
    consolidated["cac_payback_months"] = (
        consolidated["cac"]
        / (consolidated["revenue_per_deal"] * consolidated["net_margin_pct"] / 100)
    ).round(2)

    # Add time features
    consolidated["year"] = consolidated["date"].dt.year
    consolidated["month"] = consolidated["date"].dt.month
    consolidated["quarter"] = consolidated["date"].dt.quarter
    consolidated["month_name"] = consolidated["date"].dt.strftime("%b")

    print(f"✅ Merged dataset: {consolidated.shape[0]} rows, {consolidated.shape[1]} columns")

    return consolidated


def persist_to_db(
    erp_df: pd.DataFrame,
    crm_df: pd.DataFrame,
    consolidated_df: pd.DataFrame,
):
    """
    Persist data to SQLite database.

    Args:
        erp_df: Cleaned ERP data
        crm_df: Cleaned CRM data
        consolidated_df: Merged dataset
    """
    init_database()
    conn = get_db_connection()

    # Convert dates to string for SQLite
    erp_copy = erp_df.copy()
    erp_copy["date"] = erp_copy["date"].dt.strftime("%Y-%m-%d")

    crm_copy = crm_df.copy()
    crm_copy["date"] = crm_copy["date"].dt.strftime("%Y-%m-%d")

    # Insert ERP data
    erp_cols = [
        "date", "revenue", "cogs", "opex", "payroll",
        "utilities", "depreciation", "total_expenses", "net_income",
    ]
    erp_copy[erp_cols].to_sql(
        "financial_actuals", conn, if_exists="replace", index=False
    )

    # Insert CRM data
    crm_cols = [
        "date", "pipeline_value", "closed_deals", "cac",
        "conversion_rate", "avg_deal_size",
    ]
    crm_copy[crm_cols].to_sql(
        "crm_metrics", conn, if_exists="replace", index=False
    )

    conn.close()
    print(f"✅ Data persisted to SQLite: {config.DB_PATH}")


def persist_to_csv(consolidated_df: pd.DataFrame):
    """Save consolidated data to processed CSV."""
    output_path = os.path.join(
        config.PROCESSED_DATA_DIR, "consolidated_financials.csv"
    )
    consolidated_df.to_csv(output_path, index=False)
    print(f"✅ Consolidated CSV saved: {output_path}")


def load_consolidated() -> pd.DataFrame:
    """
    Load the consolidated financial dataset.

    Returns:
        pd.DataFrame with all financial and CRM metrics
    """
    path = os.path.join(
        config.PROCESSED_DATA_DIR, "consolidated_financials.csv"
    )
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def run_etl_pipeline() -> pd.DataFrame:
    """
    Execute the full ETL pipeline.

    Returns:
        Consolidated DataFrame
    """
    print("=" * 60)
    print("  DATA INTEGRATION (ETL) PIPELINE")
    print("=" * 60)

    # Step 1: Load raw data
    print("\n📥 Step 1: Loading raw data...")
    erp_df, crm_df = load_raw_data()

    # Step 2: Clean data
    print("\n🧹 Step 2: Cleaning data...")
    erp_clean = clean_dataframe(erp_df, "ERP")
    crm_clean = clean_dataframe(crm_df, "CRM")

    # Step 3: Merge data
    print("\n🔗 Step 3: Merging datasets...")
    consolidated = merge_data(erp_clean, crm_clean)

    # Step 4: Persist
    print("\n💾 Step 4: Persisting data...")
    persist_to_db(erp_clean, crm_clean, consolidated)
    persist_to_csv(consolidated)

    print(f"\n{'=' * 60}")
    print(f"  ✅ ETL Pipeline Complete!")
    print(f"{'=' * 60}")

    return consolidated


if __name__ == "__main__":
    run_etl_pipeline()
