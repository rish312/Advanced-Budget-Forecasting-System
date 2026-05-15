"""
Synthetic Financial Data Generator
====================================
Generates realistic monthly ERP and CRM data simulating 5 years of
financial history with trends, seasonality, and noise.
"""

import os
import sys
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def generate_erp_data(seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic ERP financial data.

    Produces monthly records for:
      - Revenue
      - Cost of Goods Sold (COGS)
      - Operating Expenses (OpEx)
      - Payroll
      - Utilities
      - Depreciation
      - Net Income (derived)

    Returns:
        pd.DataFrame with columns: date, revenue, cogs, opex, payroll,
        utilities, depreciation, net_income
    """
    np.random.seed(seed)

    dates = pd.date_range(
        start=config.DATA_START_DATE,
        end=config.DATA_END_DATE,
        freq=config.DATA_FREQUENCY,
    )

    n = len(dates)
    records = []

    for i, date in enumerate(dates):
        month_idx = date.month - 1  # 0-indexed
        year_offset = (date.year - pd.Timestamp(config.DATA_START_DATE).year)

        # ── Revenue ──────────────────────────────────────────────────────
        # Base + trend growth + seasonality + noise
        trend = config.REVENUE_BASE * (1 + config.REVENUE_GROWTH_RATE) ** i
        seasonal = config.SEASONAL_MULTIPLIERS[month_idx]
        noise = np.random.normal(1.0, 0.03)  # ±3% noise
        revenue = trend * seasonal * noise

        # ── COGS ─────────────────────────────────────────────────────────
        cogs_noise = np.random.normal(config.COGS_RATIO, 0.02)
        cogs = revenue * max(cogs_noise, 0.20)

        # ── Operating Expenses ───────────────────────────────────────────
        opex_trend = config.OPEX_BASE * (1 + 0.003) ** i  # Slower growth
        opex_seasonal = 1 + 0.05 * np.sin(2 * np.pi * month_idx / 12)
        opex = opex_trend * opex_seasonal * np.random.normal(1.0, 0.04)

        # ── Payroll ──────────────────────────────────────────────────────
        # Step increases annually + small monthly variation
        annual_raise = 1 + 0.04 * year_offset  # 4% annual raise
        payroll = (
            config.PAYROLL_BASE
            * annual_raise
            * np.random.normal(1.0, 0.01)
        )

        # ── Utilities ────────────────────────────────────────────────────
        # Seasonal pattern (higher in summer/winter)
        util_seasonal = 1 + 0.15 * np.cos(2 * np.pi * (month_idx - 6) / 12)
        utilities = (
            config.UTILITIES_BASE
            * util_seasonal
            * (1 + 0.02 * year_offset)
            * np.random.normal(1.0, 0.05)
        )

        # ── Depreciation ─────────────────────────────────────────────────
        # Relatively stable with occasional step-ups (capital purchases)
        step_up = 1.0
        if year_offset >= 2:
            step_up = 1.10
        if year_offset >= 4:
            step_up = 1.20
        depreciation = (
            config.DEPRECIATION_BASE
            * step_up
            * np.random.normal(1.0, 0.005)
        )

        # ── Net Income ───────────────────────────────────────────────────
        total_expenses = cogs + opex + payroll + utilities + depreciation
        net_income = revenue - total_expenses

        records.append({
            "date": date,
            "revenue": round(revenue, 2),
            "cogs": round(cogs, 2),
            "opex": round(opex, 2),
            "payroll": round(payroll, 2),
            "utilities": round(utilities, 2),
            "depreciation": round(depreciation, 2),
            "total_expenses": round(total_expenses, 2),
            "net_income": round(net_income, 2),
        })

    return pd.DataFrame(records)


def generate_crm_data(erp_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic CRM data correlated with ERP revenue.

    Produces monthly records for:
      - Sales Pipeline Value
      - Closed Deals Count
      - Customer Acquisition Cost (CAC)
      - Conversion Rate
      - Average Deal Size

    Args:
        erp_df: ERP DataFrame (used to correlate pipeline with revenue)

    Returns:
        pd.DataFrame with CRM metrics
    """
    np.random.seed(seed + 1)

    records = []
    for i, row in erp_df.iterrows():
        date = row["date"]
        month_idx = date.month - 1
        year_offset = date.year - pd.Timestamp(config.DATA_START_DATE).year

        # Pipeline is ~1.6x revenue (typical pipeline-to-close ratio)
        pipeline_value = (
            row["revenue"]
            * np.random.normal(1.6, 0.15)
        )

        # Closed deals grow with revenue trend
        deals_trend = config.CLOSED_DEALS_BASE * (1 + 0.02) ** i
        deals_seasonal = config.SEASONAL_MULTIPLIERS[month_idx]
        closed_deals = int(
            deals_trend * deals_seasonal * np.random.normal(1.0, 0.08)
        )
        closed_deals = max(closed_deals, 10)

        # CAC tends to decrease slightly over time (efficiency gains)
        cac = (
            config.CAC_BASE
            * (1 - 0.002) ** i
            * np.random.normal(1.0, 0.06)
        )

        # Conversion rate (typically 20-35%)
        conversion_rate = np.clip(
            np.random.normal(0.28, 0.04), 0.15, 0.45
        )

        # Average deal size
        avg_deal_size = (
            row["revenue"] / closed_deals if closed_deals > 0 else 0
        )

        records.append({
            "date": date,
            "pipeline_value": round(pipeline_value, 2),
            "closed_deals": closed_deals,
            "cac": round(cac, 2),
            "conversion_rate": round(conversion_rate, 4),
            "avg_deal_size": round(avg_deal_size, 2),
        })

    return pd.DataFrame(records)


def generate_all_data(seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate both ERP and CRM datasets, save to CSV.

    Returns:
        Tuple of (erp_df, crm_df)
    """
    print("=" * 60)
    print("  SYNTHETIC FINANCIAL DATA GENERATOR")
    print("=" * 60)

    # Generate ERP data
    print(f"\n📊 Generating ERP data ({config.NUM_MONTHS} months)...")
    erp_df = generate_erp_data(seed)
    erp_path = os.path.join(config.RAW_DATA_DIR, "erp_financials.csv")
    erp_df.to_csv(erp_path, index=False)
    print(f"   ✅ Saved to {erp_path}")
    print(f"   Revenue range: ${erp_df['revenue'].min():,.0f} — ${erp_df['revenue'].max():,.0f}")
    print(f"   Net Income range: ${erp_df['net_income'].min():,.0f} — ${erp_df['net_income'].max():,.0f}")

    # Generate CRM data
    print(f"\n📈 Generating CRM data ({config.NUM_MONTHS} months)...")
    crm_df = generate_crm_data(erp_df, seed)
    crm_path = os.path.join(config.RAW_DATA_DIR, "crm_metrics.csv")
    crm_df.to_csv(crm_path, index=False)
    print(f"   ✅ Saved to {crm_path}")
    print(f"   Closed deals range: {crm_df['closed_deals'].min()} — {crm_df['closed_deals'].max()}")
    print(f"   CAC range: ${crm_df['cac'].min():,.0f} — ${crm_df['cac'].max():,.0f}")

    print(f"\n{'=' * 60}")
    print(f"  ✅ Data generation complete! ({len(erp_df)} records each)")
    print(f"{'=' * 60}")

    return erp_df, crm_df


if __name__ == "__main__":
    generate_all_data()
