"""
Model Evaluation Module
========================
Standardized evaluation metrics and comparison utilities.
"""
import pandas as pd
import numpy as np


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Calculate MAE, RMSE, MAPE, R²."""
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    return {"mae": round(mae, 2), "rmse": round(rmse, 2),
            "mape": round(mape, 2), "r_squared": round(r2, 4)}


def compare_models(metrics_list: list[dict]) -> pd.DataFrame:
    """Create comparison table from list of model metric dicts."""
    df = pd.DataFrame(metrics_list)
    if "mape" in df.columns:
        df = df.sort_values("mape")
    return df


def format_metrics_report(metrics: dict) -> str:
    """Pretty-print a metrics dictionary."""
    lines = [f"{'Metric':<15} {'Value':>12}"]
    lines.append("-" * 28)
    for k, v in metrics.items():
        if k == "model":
            continue
        lines.append(f"{k:<15} {str(v):>12}")
    return "\n".join(lines)
