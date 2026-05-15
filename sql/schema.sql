-- Budget Forecasting System — SQLite Schema
-- ============================================
-- Creates the database schema for financial data storage,
-- forecast results, and variance tracking.

-- ─── Financial Actuals (from ERP) ──────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS financial_actuals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    date            TEXT NOT NULL,           -- YYYY-MM-DD (month start)
    revenue         REAL NOT NULL,
    cogs            REAL NOT NULL,
    opex            REAL NOT NULL,
    payroll         REAL NOT NULL,
    utilities       REAL NOT NULL,
    depreciation    REAL NOT NULL,
    total_expenses  REAL NOT NULL,
    net_income      REAL NOT NULL,
    created_at      TEXT DEFAULT (datetime('now')),

    UNIQUE(date)
);

-- ─── CRM Metrics ───────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS crm_metrics (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    date            TEXT NOT NULL,           -- YYYY-MM-DD (month start)
    pipeline_value  REAL NOT NULL,
    closed_deals    INTEGER NOT NULL,
    cac             REAL NOT NULL,
    conversion_rate REAL NOT NULL,
    avg_deal_size   REAL NOT NULL,
    created_at      TEXT DEFAULT (datetime('now')),

    UNIQUE(date)
);

-- ─── Consolidated View (joined source of truth) ────────────────────────────────
CREATE VIEW IF NOT EXISTS v_consolidated_financials AS
SELECT
    f.date,
    f.revenue,
    f.cogs,
    f.opex,
    f.payroll,
    f.utilities,
    f.depreciation,
    f.total_expenses,
    f.net_income,
    c.pipeline_value,
    c.closed_deals,
    c.cac,
    c.conversion_rate,
    c.avg_deal_size,
    -- Derived metrics
    ROUND(f.cogs / f.revenue * 100, 2)             AS cogs_pct,
    ROUND(f.net_income / f.revenue * 100, 2)        AS net_margin_pct,
    ROUND(f.revenue / c.closed_deals, 2)            AS revenue_per_deal
FROM financial_actuals f
LEFT JOIN crm_metrics c ON f.date = c.date
ORDER BY f.date;

-- ─── Forecasts ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS forecasts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name      TEXT NOT NULL,           -- 'arima', 'prophet', 'exp_smoothing', 'ensemble'
    target_metric   TEXT NOT NULL,           -- 'revenue', 'net_income', etc.
    forecast_date   TEXT NOT NULL,           -- Date being forecasted
    forecast_value  REAL NOT NULL,
    lower_ci        REAL,                    -- Lower confidence interval
    upper_ci        REAL,                    -- Upper confidence interval
    run_date        TEXT DEFAULT (datetime('now')),

    UNIQUE(model_name, target_metric, forecast_date, run_date)
);

-- ─── Variance Log ──────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS variance_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    date            TEXT NOT NULL,
    metric          TEXT NOT NULL,           -- 'revenue', 'net_income', etc.
    actual_value    REAL NOT NULL,
    forecast_value  REAL NOT NULL,
    variance        REAL NOT NULL,           -- actual - forecast
    variance_pct    REAL NOT NULL,           -- (actual - forecast) / forecast * 100
    classification  TEXT NOT NULL,           -- 'Favorable' or 'Adverse'
    model_name      TEXT NOT NULL,
    created_at      TEXT DEFAULT (datetime('now'))
);

-- ─── Indexes for Performance ────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_actuals_date ON financial_actuals(date);
CREATE INDEX IF NOT EXISTS idx_crm_date ON crm_metrics(date);
CREATE INDEX IF NOT EXISTS idx_forecasts_model_metric ON forecasts(model_name, target_metric);
CREATE INDEX IF NOT EXISTS idx_forecasts_date ON forecasts(forecast_date);
CREATE INDEX IF NOT EXISTS idx_variance_date ON variance_log(date);
CREATE INDEX IF NOT EXISTS idx_variance_metric ON variance_log(metric);
