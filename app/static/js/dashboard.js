/* ═══════════════════════════════════════════════════════════════════════
   Budget Forecasting Dashboard — Interactive Charts (Plotly.js)
   ═══════════════════════════════════════════════════════════════════════ */

// ─── Plotly Theme ──────────────────────────────────────────────────────
const PLOTLY_LAYOUT = {
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    font: { family: 'Inter, sans-serif', color: '#94a3b8', size: 12 },
    margin: { t: 10, r: 20, b: 40, l: 60 },
    xaxis: {
        gridcolor: 'rgba(255,255,255,0.04)',
        linecolor: 'rgba(255,255,255,0.06)',
        tickfont: { size: 11 },
    },
    yaxis: {
        gridcolor: 'rgba(255,255,255,0.04)',
        linecolor: 'rgba(255,255,255,0.06)',
        tickfont: { size: 11 },
    },
    legend: { orientation: 'h', y: -0.15, font: { size: 11 } },
    hoverlabel: {
        bgcolor: '#1e293b',
        bordercolor: '#6366f1',
        font: { family: 'Inter', color: '#f1f5f9', size: 13 },
    },
};

const PLOTLY_CONFIG = { responsive: true, displayModeBar: false };

const COLORS = {
    blue: '#6366f1', cyan: '#06b6d4', green: '#10b981',
    amber: '#f59e0b', rose: '#f43f5e', purple: '#a855f7',
    slate: '#64748b',
};

// ─── Utility: Format currency ──────────────────────────────────────────
function fmtCurrency(val) {
    if (Math.abs(val) >= 1e6) return '$' + (val / 1e6).toFixed(1) + 'M';
    if (Math.abs(val) >= 1e3) return '$' + (val / 1e3).toFixed(0) + 'K';
    return '$' + val.toFixed(0);
}

// ─── Dashboard Charts ──────────────────────────────────────────────────
async function loadDashboardCharts() {
    try {
        const res = await fetch('/api/financial-data');
        const data = await res.json();
        if (data.error) return;

        renderRevenueTrend(data);
        renderExpensePie(data);
        renderCashFlow(data);
        loadAnnualChart();
    } catch (e) { console.error('Dashboard chart error:', e); }
}

function renderRevenueTrend(data) {
    const dates = data.map(d => d.date);
    const revenue = data.map(d => d.revenue);
    const netIncome = data.map(d => d.net_income);

    const traces = [
        {
            x: dates, y: revenue, name: 'Revenue', type: 'scatter', mode: 'lines',
            line: { color: COLORS.blue, width: 2.5 },
            fill: 'tonexty', fillcolor: 'rgba(99,102,241,0.08)',
        },
        {
            x: dates, y: netIncome, name: 'Net Income', type: 'scatter', mode: 'lines',
            line: { color: COLORS.green, width: 2 },
            fill: 'tozeroy', fillcolor: 'rgba(16,185,129,0.06)',
        },
    ];

    const layout = {
        ...PLOTLY_LAYOUT,
        yaxis: { ...PLOTLY_LAYOUT.yaxis, tickprefix: '$', tickformat: ',.0f' },
        showlegend: true,
    };

    Plotly.newPlot('chart-revenue-trend', traces, layout, PLOTLY_CONFIG);
}

function renderExpensePie(data) {
    const latest = data[data.length - 1];
    const labels = ['COGS', 'OpEx', 'Payroll', 'Utilities', 'Depreciation'];
    const values = [latest.cogs, latest.opex, latest.payroll, latest.utilities, latest.depreciation];
    const colors = [COLORS.rose, COLORS.blue, COLORS.green, COLORS.amber, COLORS.purple];

    const trace = {
        labels, values, type: 'pie', hole: 0.55,
        marker: { colors },
        textinfo: 'percent', textfont: { color: '#f1f5f9', size: 12 },
        hoverinfo: 'label+value+percent',
        hoverlabel: PLOTLY_LAYOUT.hoverlabel,
    };

    const layout = {
        ...PLOTLY_LAYOUT,
        margin: { t: 10, r: 10, b: 10, l: 10 },
        showlegend: true,
        legend: { orientation: 'h', y: -0.1, font: { size: 11, color: '#94a3b8' } },
        annotations: [{
            text: fmtCurrency(latest.total_expenses),
            font: { size: 16, color: '#f1f5f9', family: 'Inter' },
            showarrow: false,
        }],
    };

    Plotly.newPlot('chart-expense-pie', [trace], layout, PLOTLY_CONFIG);
}

function renderCashFlow(data) {
    const dates = data.map(d => d.date);
    const cashFlow = data.map(d => d.net_income);
    const colors = cashFlow.map(v => v >= 0 ? COLORS.green : COLORS.rose);

    const trace = {
        x: dates, y: cashFlow, type: 'bar',
        marker: { color: colors, opacity: 0.8 },
        name: 'Net Cash Flow',
    };

    const layout = {
        ...PLOTLY_LAYOUT,
        yaxis: { ...PLOTLY_LAYOUT.yaxis, tickprefix: '$', tickformat: ',.0f' },
        bargap: 0.3,
    };

    Plotly.newPlot('chart-cashflow', [trace], layout, PLOTLY_CONFIG);
}

async function loadAnnualChart() {
    try {
        const res = await fetch('/api/kpi-summary');
        const data = await res.json();
        if (data.error) return;

        const years = data.map(d => d.year.toString());
        const traces = [
            {
                x: years, y: data.map(d => d.revenue), name: 'Revenue', type: 'bar',
                marker: { color: COLORS.blue, opacity: 0.85 },
            },
            {
                x: years, y: data.map(d => d.total_expenses), name: 'Expenses', type: 'bar',
                marker: { color: COLORS.rose, opacity: 0.7 },
            },
            {
                x: years, y: data.map(d => d.net_income), name: 'Net Income', type: 'bar',
                marker: { color: COLORS.green, opacity: 0.85 },
            },
        ];

        const layout = {
            ...PLOTLY_LAYOUT,
            barmode: 'group', bargap: 0.2, bargroupgap: 0.1,
            yaxis: { ...PLOTLY_LAYOUT.yaxis, tickprefix: '$', tickformat: ',.0f' },
        };

        Plotly.newPlot('chart-annual', traces, layout, PLOTLY_CONFIG);
    } catch (e) { console.error('Annual chart error:', e); }
}

// ─── Forecast Charts ───────────────────────────────────────────────────
async function loadForecastCharts() {
    try {
        const [histRes, fcstRes, metricsRes] = await Promise.all([
            fetch('/api/financial-data'),
            fetch('/api/forecast-data'),
            fetch('/api/metrics-data'),
        ]);

        const histData = await histRes.json();
        const fcstData = await fcstRes.json();
        const metricsData = await metricsRes.json();

        renderForecastComparison(histData, fcstData);
        if (!metricsData.error) {
            renderModelMetrics(metricsData);
            renderMetricsTable(metricsData);
        }
    } catch (e) { console.error('Forecast chart error:', e); }
}

function renderForecastComparison(histData, fcstData) {
    const el = document.getElementById('chart-forecast');
    if (!el) return;

    const selectedModel = document.getElementById('model-select')?.value || 'all';

    const traces = [];

    // Historical data
    if (!histData.error) {
        traces.push({
            x: histData.map(d => d.date),
            y: histData.map(d => d.revenue),
            name: 'Historical Revenue',
            type: 'scatter', mode: 'lines',
            line: { color: '#94a3b8', width: 2 },
        });
    }

    if (fcstData.error) {
        Plotly.newPlot(el, traces, { ...PLOTLY_LAYOUT }, PLOTLY_CONFIG);
        return;
    }

    const modelColors = {
        arima: COLORS.blue, prophet: COLORS.cyan,
        exp_smoothing: COLORS.amber, ensemble: COLORS.green,
    };

    const models = [...new Set(fcstData.map(d => d.model))];
    models.forEach(model => {
        if (selectedModel !== 'all' && model !== selectedModel) return;

        const mData = fcstData.filter(d => d.model === model);
        const color = modelColors[model] || COLORS.slate;

        traces.push({
            x: mData.map(d => d.date),
            y: mData.map(d => d.forecast),
            name: model.toUpperCase(),
            type: 'scatter', mode: 'lines+markers',
            line: { color, width: 2.5 },
            marker: { size: 5 },
        });

        // Confidence interval
        traces.push({
            x: [...mData.map(d => d.date), ...mData.map(d => d.date).reverse()],
            y: [...mData.map(d => d.upper_ci), ...mData.map(d => d.lower_ci).reverse()],
            fill: 'toself', fillcolor: color.replace(')', ',0.1)').replace('rgb', 'rgba'),
            line: { color: 'transparent' },
            name: model + ' CI', showlegend: false, type: 'scatter',
        });
    });

    const layout = {
        ...PLOTLY_LAYOUT,
        yaxis: { ...PLOTLY_LAYOUT.yaxis, tickprefix: '$', tickformat: ',.0f' },
        showlegend: true,
    };

    Plotly.newPlot(el, traces, layout, PLOTLY_CONFIG);
}

function renderModelMetrics(data) {
    const el = document.getElementById('chart-model-metrics');
    if (!el || !data.length) return;

    const models = data.map(d => (d.model || '').toUpperCase());
    const mape = data.map(d => d.mape || 0);
    const rmse = data.map(d => d.rmse || 0);

    const traces = [
        {
            x: models, y: mape, name: 'MAPE (%)', type: 'bar',
            marker: { color: COLORS.blue, opacity: 0.85 },
        },
    ];

    Plotly.newPlot(el, traces, { ...PLOTLY_LAYOUT, barmode: 'group' }, PLOTLY_CONFIG);
}

function renderMetricsTable(data) {
    const container = document.getElementById('metrics-table-container');
    if (!container || !data.length) return;

    let html = '<table class="data-table"><thead><tr>';
    html += '<th>Model</th><th>MAE</th><th>RMSE</th><th>MAPE (%)</th><th>R²</th>';
    html += '</tr></thead><tbody>';

    data.forEach(d => {
        html += '<tr>';
        html += `<td style="color:#f1f5f9;font-weight:600">${(d.model||'').toUpperCase()}</td>`;
        html += `<td>${fmtCurrency(d.mae || 0)}</td>`;
        html += `<td>${fmtCurrency(d.rmse || 0)}</td>`;
        html += `<td>${(d.mape || 0).toFixed(2)}%</td>`;
        html += `<td>${(d.r_squared || 0).toFixed(4)}</td>`;
        html += '</tr>';
    });

    html += '</tbody></table>';
    container.innerHTML = html;
}

// ─── Variance Charts ───────────────────────────────────────────────────
async function loadVarianceCharts() {
    try {
        const res = await fetch('/api/variance-data');
        const data = await res.json();
        if (data.error) {
            document.getElementById('chart-variance-time').innerHTML =
                '<div class="empty-state"><div class="empty-icon">📭</div><h2>No Variance Data</h2><p>Run the pipeline first.</p></div>';
            return;
        }

        renderVarianceTimeline(data);
        renderWaterfall(data);
        renderVarianceDist(data);
        renderVarianceTable(data);
    } catch (e) { console.error('Variance chart error:', e); }
}

function renderVarianceTimeline(data) {
    const el = document.getElementById('chart-variance-time');
    if (!el) return;

    const colors = data.map(d =>
        d.classification === 'Favorable' ? COLORS.green : COLORS.rose
    );

    const traces = [
        {
            x: data.map(d => d.date),
            y: data.map(d => d.variance_pct),
            type: 'bar',
            marker: { color: colors, opacity: 0.85 },
            name: 'Variance %',
            hovertemplate: '%{x}<br>Variance: %{y:.1f}%<extra></extra>',
        },
        {
            x: data.map(d => d.date),
            y: data.map(() => 0),
            type: 'scatter', mode: 'lines',
            line: { color: '#64748b', width: 1, dash: 'dot' },
            showlegend: false,
        },
    ];

    Plotly.newPlot(el, traces, {
        ...PLOTLY_LAYOUT,
        yaxis: { ...PLOTLY_LAYOUT.yaxis, ticksuffix: '%' },
    }, PLOTLY_CONFIG);
}

function renderWaterfall(data) {
    const el = document.getElementById('chart-waterfall');
    if (!el) return;

    const last6 = data.slice(-6);
    const trace = {
        type: 'waterfall',
        x: last6.map(d => d.date.substring(0, 7)),
        y: last6.map(d => d.variance),
        measure: last6.map(() => 'relative'),
        connector: { line: { color: 'rgba(255,255,255,0.1)' } },
        decreasing: { marker: { color: COLORS.rose } },
        increasing: { marker: { color: COLORS.green } },
        totals: { marker: { color: COLORS.blue } },
    };

    Plotly.newPlot(el, [trace], {
        ...PLOTLY_LAYOUT,
        yaxis: { ...PLOTLY_LAYOUT.yaxis, tickprefix: '$' },
    }, PLOTLY_CONFIG);
}

function renderVarianceDist(data) {
    const el = document.getElementById('chart-variance-dist');
    if (!el) return;

    const trace = {
        x: data.map(d => d.variance_pct),
        type: 'histogram', nbinsx: 15,
        marker: { color: COLORS.blue, opacity: 0.75 },
        name: 'Variance %',
    };

    Plotly.newPlot(el, [trace], {
        ...PLOTLY_LAYOUT,
        xaxis: { ...PLOTLY_LAYOUT.xaxis, title: 'Variance (%)' },
        yaxis: { ...PLOTLY_LAYOUT.yaxis, title: 'Frequency' },
    }, PLOTLY_CONFIG);
}

function renderVarianceTable(data) {
    const container = document.getElementById('variance-table-container');
    if (!container) return;

    let html = '<table class="data-table"><thead><tr>';
    html += '<th>Date</th><th>Actual</th><th>Forecast</th><th>Variance</th><th>Var %</th><th>Status</th>';
    html += '</tr></thead><tbody>';

    data.slice(-12).reverse().forEach(d => {
        const cls = d.classification === 'Favorable' ? 'badge-favorable' : 'badge-adverse';
        html += '<tr>';
        html += `<td style="color:#f1f5f9">${d.date.substring(0, 10)}</td>`;
        html += `<td>${fmtCurrency(d.actual_value)}</td>`;
        html += `<td>${fmtCurrency(d.forecast_value)}</td>`;
        html += `<td>${fmtCurrency(d.variance)}</td>`;
        html += `<td>${d.variance_pct.toFixed(1)}%</td>`;
        html += `<td><span class="badge ${cls}">${d.classification}</span></td>`;
        html += '</tr>';
    });

    html += '</tbody></table>';
    container.innerHTML = html;
}

// ─── Filter Buttons ────────────────────────────────────────────────────
document.querySelectorAll('.btn-filter').forEach(btn => {
    btn.addEventListener('click', function() {
        document.querySelectorAll('.btn-filter').forEach(b => b.classList.remove('active'));
        this.classList.add('active');
    });
});
