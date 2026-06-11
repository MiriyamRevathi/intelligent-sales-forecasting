import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder


class InventoryOptimizationError(Exception):
    """Raised when inventory optimization cannot be completed."""


FORECAST_ALIASES = ("forecast_revenue", "forecast_sales", "forecast", "predicted_revenue", "predicted_sales")
DATE_ALIASES = ("date", "forecast_date", "day")
PLOTLY_TEMPLATE = "plotly_dark"
CHART_HEIGHT = 390


def normalize_name(name):
    return str(name).strip().lower().replace(" ", "_")


def find_column(df, aliases):
    normalized = {normalize_name(column): column for column in df.columns}
    for alias in aliases:
        if alias in normalized:
            return normalized[alias]
    return None


def load_forecast_results(report_folder):
    path = Path(report_folder) / "forecast_results.csv"
    if not path.exists():
        raise InventoryOptimizationError("Missing forecast file. Generate a sales forecast first.")

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise InventoryOptimizationError("The forecast results file is empty.") from exc
    except Exception as exc:
        raise InventoryOptimizationError("The forecast results file could not be read.") from exc

    if df.empty:
        raise InventoryOptimizationError("The forecast results file is empty.")

    forecast_column = find_column(df, FORECAST_ALIASES)
    if not forecast_column:
        raise InventoryOptimizationError("Missing forecast demand column in reports/forecast_results.csv.")

    df[forecast_column] = pd.to_numeric(df[forecast_column], errors="coerce")
    df = df.dropna(subset=[forecast_column])
    if df.empty:
        raise InventoryOptimizationError("Forecast results do not contain valid numeric demand values.")

    date_column = find_column(df, DATE_ALIASES)
    if date_column:
        df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
        df = df.sort_values(date_column)

    return df, path, forecast_column, date_column


def validate_inputs(current_stock, lead_time_days):
    try:
        stock = float(current_stock)
    except (TypeError, ValueError) as exc:
        raise InventoryOptimizationError("Current stock must be a valid number.") from exc

    try:
        lead_time = int(lead_time_days)
    except (TypeError, ValueError) as exc:
        raise InventoryOptimizationError("Lead time must be a valid whole number of days.") from exc

    if stock < 0:
        raise InventoryOptimizationError("Current stock cannot be negative.")
    if lead_time <= 0:
        raise InventoryOptimizationError("Lead time must be greater than zero.")

    return stock, lead_time


def classify_inventory(current_stock, reorder_point, forecast_demand, days_until_stockout, lead_time_days):
    if current_stock <= 0 or days_until_stockout <= lead_time_days * 0.5:
        return "Critical Stock", "danger"
    if current_stock <= reorder_point or days_until_stockout <= lead_time_days:
        return "Low Stock", "warning"
    if forecast_demand > 0 and current_stock >= forecast_demand * 1.5:
        return "Overstock", "info"
    return "Healthy", "success"


def generate_alerts(classification, current_stock, reorder_point, forecast_demand):
    alerts = []

    if current_stock <= reorder_point:
        alerts.append(
            {
                "title": "Reorder Required",
                "message": "Current stock is at or below the reorder point.",
                "level": "warning",
                "icon": "bi-cart-plus",
            }
        )

    if classification == "Critical Stock":
        alerts.append(
            {
                "title": "Critical Inventory Warning",
                "message": "Stock may run out before replenishment arrives.",
                "level": "danger",
                "icon": "bi-exclamation-octagon",
            }
        )

    if classification == "Overstock":
        alerts.append(
            {
                "title": "Overstock Warning",
                "message": "Current stock is significantly higher than forecast demand.",
                "level": "info",
                "icon": "bi-box-seam",
            }
        )

    if not alerts:
        alerts.append(
            {
                "title": "Inventory Healthy",
                "message": "Current stock is aligned with forecast demand and lead time.",
                "level": "success",
                "icon": "bi-check-circle",
            }
        )

    return alerts


def apply_chart_layout(fig):
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=CHART_HEIGHT,
        paper_bgcolor="#171b24",
        plot_bgcolor="#171b24",
        margin=dict(l=40, r=24, t=48, b=44),
        font=dict(color="#f4f7fb"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.15)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.15)")
    return fig


def chart_payload(fig):
    return json.loads(json.dumps(fig, cls=PlotlyJSONEncoder))


def build_charts(forecast_df, forecast_column, date_column, current_stock, reorder_point, safety_stock, forecast_demand, classification):
    charts = []
    health_score = max(0, min(100, (current_stock / reorder_point) * 100 if reorder_point > 0 else 100))

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=health_score,
            number={"suffix": "%"},
            title={"text": f"Inventory Health - {classification}"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#66d9ef"},
                "steps": [
                    {"range": [0, 40], "color": "#842029"},
                    {"range": [40, 75], "color": "#664d03"},
                    {"range": [75, 100], "color": "#0f5132"},
                ],
                "threshold": {"line": {"color": "#ffffff", "width": 3}, "thickness": 0.75, "value": 75},
            },
        )
    )
    charts.append({"id": "inventoryHealthGauge", "title": "Inventory Health Gauge", "figure": chart_payload(apply_chart_layout(fig))})

    comparison = pd.DataFrame(
        {
            "Metric": ["Current Stock", "Forecast Demand", "Safety Stock"],
            "Units": [current_stock, forecast_demand, safety_stock],
        }
    )
    fig = px.bar(comparison, x="Metric", y="Units", title="Stock vs Forecast Demand")
    charts.append({"id": "stockVsForecast", "title": "Stock vs Forecast Demand Chart", "figure": chart_payload(apply_chart_layout(fig))})

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Current Stock", x=["Inventory"], y=[current_stock], marker_color="#66d9ef"))
    fig.add_trace(go.Scatter(name="Reorder Point", x=["Inventory"], y=[reorder_point], mode="markers", marker={"size": 18, "color": "#ffc107", "symbol": "line-ew"}))
    fig.add_trace(go.Scatter(name="Safety Stock", x=["Inventory"], y=[safety_stock], mode="markers", marker={"size": 16, "color": "#dc3545", "symbol": "line-ew"}))
    fig.update_layout(title="Reorder Point Indicator", yaxis_title="Units")
    charts.append({"id": "reorderPointIndicator", "title": "Reorder Point Indicator", "figure": chart_payload(apply_chart_layout(fig))})

    risk_df = forecast_df.copy()
    risk_df["risk"] = pd.cut(
        risk_df[forecast_column],
        bins=[-float("inf"), safety_stock, reorder_point, float("inf")],
        labels=["Low Demand Risk", "Moderate Demand Risk", "High Demand Risk"],
    )
    risk_counts = risk_df["risk"].value_counts().reset_index()
    risk_counts.columns = ["Risk", "Days"]
    fig = px.pie(risk_counts, names="Risk", values="Days", title="Inventory Risk Distribution", hole=0.45)
    charts.append({"id": "inventoryRiskDistribution", "title": "Inventory Risk Distribution", "figure": chart_payload(apply_chart_layout(fig))})

    if date_column:
        trend = forecast_df[[date_column, forecast_column]].copy()
        trend["Current Stock"] = current_stock
        trend["Reorder Point"] = reorder_point
        # Keep this data available for the chart grid through the existing stock-vs-demand visual set.

    return charts


def money_or_units(value):
    return f"{float(value):,.2f}"


def save_inventory_report(report, report_folder):
    output_path = Path(report_folder) / "inventory_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return output_path


def optimize_inventory(report_folder, current_stock, lead_time_days):
    stock, lead_time = validate_inputs(current_stock, lead_time_days)
    df, forecast_path, forecast_column, date_column = load_forecast_results(report_folder)

    demand = df[forecast_column].clip(lower=0)
    average_daily_demand = float(demand.mean())
    maximum_daily_demand = float(demand.max())
    forecast_demand = float(demand.sum())

    if average_daily_demand <= 0:
        raise InventoryOptimizationError("Average daily forecast demand must be greater than zero.")

    safety_stock = max((maximum_daily_demand * lead_time) - (average_daily_demand * lead_time), 0)
    reorder_point = (average_daily_demand * lead_time) + safety_stock
    recommended_order_quantity = max(forecast_demand - stock, 0)
    days_until_stockout = stock / average_daily_demand

    classification, classification_level = classify_inventory(
        stock,
        reorder_point,
        forecast_demand,
        days_until_stockout,
        lead_time,
    )
    alerts = generate_alerts(classification, stock, reorder_point, forecast_demand)
    charts = build_charts(
        df,
        forecast_column,
        date_column,
        stock,
        reorder_point,
        safety_stock,
        forecast_demand,
        classification,
    )

    report = {
        "source_file": forecast_path.name,
        "current_stock": round(stock, 2),
        "lead_time_days": lead_time,
        "average_daily_demand": round(average_daily_demand, 2),
        "maximum_daily_demand": round(maximum_daily_demand, 2),
        "forecast_demand": round(forecast_demand, 2),
        "safety_stock": round(safety_stock, 2),
        "reorder_point": round(reorder_point, 2),
        "recommended_order_quantity": round(recommended_order_quantity, 2),
        "days_until_stockout": round(days_until_stockout, 2),
        "classification": classification,
        "classification_level": classification_level,
        "alerts": alerts,
    }

    output_path = save_inventory_report(report, report_folder)

    kpis = [
        {"label": "Current Stock", "value": money_or_units(stock), "icon": "bi-boxes"},
        {"label": "Safety Stock", "value": money_or_units(safety_stock), "icon": "bi-shield-check"},
        {"label": "Reorder Point", "value": money_or_units(reorder_point), "icon": "bi-signpost-split"},
        {"label": "Recommended Order Quantity", "value": money_or_units(recommended_order_quantity), "icon": "bi-cart-plus"},
        {"label": "Days Until Stockout", "value": money_or_units(days_until_stockout), "icon": "bi-hourglass-split"},
    ]

    return {
        "report": report,
        "report_filename": output_path.name,
        "kpis": kpis,
        "charts": charts,
    }
