"""
Exploratory Data Analysis (EDA) Module
=======================================
Generates comprehensive visualizations and statistical summaries
for the consolidated financial dataset.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.utils import format_currency

# Set style
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")
FIGSIZE = (14, 6)
DPI = 150


def plot_revenue_trend(df: pd.DataFrame):
    """Plot revenue over time with trend line and YoY growth."""
    fig, ax1 = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    # Revenue line
    ax1.plot(
        df["date"], df["revenue"], color="#2196F3", linewidth=2,
        marker="o", markersize=3, label="Monthly Revenue"
    )

    # 3-month moving average
    ma3 = df["revenue"].rolling(3).mean()
    ax1.plot(
        df["date"], ma3, color="#FF9800", linewidth=2,
        linestyle="--", label="3-Month MA"
    )

    # 12-month moving average
    ma12 = df["revenue"].rolling(12).mean()
    ax1.plot(
        df["date"], ma12, color="#4CAF50", linewidth=2.5,
        linestyle="-.", label="12-Month MA"
    )

    ax1.set_title("Revenue Trend Analysis", fontsize=16, fontweight="bold", pad=15)
    ax1.set_xlabel("Date", fontsize=12)
    ax1.set_ylabel("Revenue ($)", fontsize=12)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
    ax1.legend(loc="upper left", fontsize=10)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(config.FIGURES_DIR, "revenue_trend.png"), bbox_inches="tight")
    plt.close()
    print("   📊 Saved revenue_trend.png")


def plot_expense_breakdown(df: pd.DataFrame):
    """Stacked area chart of expense categories over time."""
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    expense_cols = ["cogs", "opex", "payroll", "utilities", "depreciation"]
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]

    ax.stackplot(
        df["date"],
        *[df[col] for col in expense_cols],
        labels=["COGS", "OpEx", "Payroll", "Utilities", "Depreciation"],
        colors=colors,
        alpha=0.8,
    )

    ax.set_title("Expense Breakdown Over Time", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Expenses ($)", fontsize=12)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(config.FIGURES_DIR, "expense_breakdown.png"), bbox_inches="tight")
    plt.close()
    print("   📊 Saved expense_breakdown.png")


def plot_profitability(df: pd.DataFrame):
    """Revenue vs Net Income with margin trend."""
    fig, ax1 = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    ax1.bar(df["date"], df["revenue"], color="#2196F3", alpha=0.4, label="Revenue", width=20)
    ax1.bar(df["date"], df["net_income"], color="#4CAF50", alpha=0.6, label="Net Income", width=20)
    ax1.set_ylabel("Amount ($)", fontsize=12)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))

    ax2 = ax1.twinx()
    ax2.plot(
        df["date"], df["net_margin_pct"], color="#FF5722",
        linewidth=2.5, label="Net Margin %"
    )
    ax2.set_ylabel("Net Margin (%)", fontsize=12, color="#FF5722")
    ax2.tick_params(axis="y", labelcolor="#FF5722")

    ax1.set_title("Profitability Analysis", fontsize=16, fontweight="bold", pad=15)
    ax1.set_xlabel("Date", fontsize=12)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=10)
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(config.FIGURES_DIR, "profitability.png"), bbox_inches="tight")
    plt.close()
    print("   📊 Saved profitability.png")


def plot_seasonal_decomposition(df: pd.DataFrame, column: str = "revenue"):
    """Decompose time series into trend, seasonal, and residual components."""
    ts = df.set_index("date")[column]
    ts.index = pd.DatetimeIndex(ts.index, freq=config.DATA_FREQUENCY)

    # Use additive if data contains zero or negative values
    decomp_model = "multiplicative" if (ts > 0).all() else "additive"
    result = seasonal_decompose(ts, model=decomp_model, period=12)

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), dpi=DPI, sharex=True)

    result.observed.plot(ax=axes[0], color="#2196F3", linewidth=1.5)
    axes[0].set_ylabel("Observed", fontsize=11)
    axes[0].set_title(
        f"Seasonal Decomposition — {column.title()}",
        fontsize=16, fontweight="bold", pad=15,
    )

    result.trend.plot(ax=axes[1], color="#4CAF50", linewidth=2)
    axes[1].set_ylabel("Trend", fontsize=11)

    result.seasonal.plot(ax=axes[2], color="#FF9800", linewidth=1.5)
    axes[2].set_ylabel("Seasonal", fontsize=11)

    result.resid.plot(ax=axes[3], color="#F44336", linewidth=1, marker="o", markersize=2)
    axes[3].set_ylabel("Residual", fontsize=11)

    for ax in axes:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(config.FIGURES_DIR, f"decomposition_{column}.png"),
        bbox_inches="tight",
    )
    plt.close()
    print(f"   📊 Saved decomposition_{column}.png")


def plot_correlation_heatmap(df: pd.DataFrame):
    """Correlation heatmap of key financial metrics."""
    metrics = [
        "revenue", "cogs", "opex", "payroll", "net_income",
        "pipeline_value", "closed_deals", "cac", "gross_margin_pct",
    ]
    available = [m for m in metrics if m in df.columns]
    corr = df[available].corr()

    fig, ax = plt.subplots(figsize=(10, 8), dpi=DPI)
    mask = np.triu(np.ones_like(corr, dtype=bool))

    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="RdYlBu_r", center=0, square=True,
        linewidths=1, linecolor="white",
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )

    ax.set_title(
        "Financial Metrics Correlation",
        fontsize=16, fontweight="bold", pad=15,
    )

    plt.tight_layout()
    plt.savefig(os.path.join(config.FIGURES_DIR, "correlation_heatmap.png"), bbox_inches="tight")
    plt.close()
    print("   📊 Saved correlation_heatmap.png")


def plot_monthly_distribution(df: pd.DataFrame):
    """Box plots showing monthly distribution of revenue."""
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    sns.boxplot(
        data=df, x="month_name", y="revenue",
        order=month_order, palette="viridis", ax=ax,
    )

    ax.set_title("Revenue Distribution by Month", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel("Revenue ($)", fontsize=12)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(config.FIGURES_DIR, "monthly_distribution.png"), bbox_inches="tight")
    plt.close()
    print("   📊 Saved monthly_distribution.png")


def generate_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Generate comprehensive summary statistics."""
    numeric_cols = [
        "revenue", "cogs", "opex", "payroll", "utilities",
        "depreciation", "total_expenses", "net_income",
        "pipeline_value", "closed_deals", "cac",
    ]
    available = [c for c in numeric_cols if c in df.columns]

    summary = df[available].describe().round(2)

    # Add additional stats
    summary.loc["median"] = df[available].median().round(2)
    summary.loc["skew"] = df[available].skew().round(4)
    summary.loc["kurtosis"] = df[available].kurtosis().round(4)

    # Save
    summary_path = os.path.join(config.REPORTS_DIR, "summary_statistics.csv")
    summary.to_csv(summary_path)
    print(f"   📄 Saved summary statistics to {summary_path}")

    return summary


def run_eda(df: pd.DataFrame):
    """Run the full EDA pipeline."""
    print("=" * 60)
    print("  EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    print("\n📊 Generating visualizations...")
    plot_revenue_trend(df)
    plot_expense_breakdown(df)
    plot_profitability(df)
    plot_seasonal_decomposition(df, "revenue")
    plot_seasonal_decomposition(df, "net_income")
    plot_correlation_heatmap(df)
    plot_monthly_distribution(df)

    print("\n📈 Generating summary statistics...")
    summary = generate_summary_statistics(df)

    print(f"\n{'=' * 60}")
    print(f"  ✅ EDA Complete! Figures saved to {config.FIGURES_DIR}")
    print(f"{'=' * 60}")

    return summary


if __name__ == "__main__":
    from src.data_integration import load_consolidated
    df = load_consolidated()
    run_eda(df)
