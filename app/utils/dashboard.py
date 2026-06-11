import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder


REVENUE_ALIASES = ("revenue", "sales", "total_sales", "sales_amount", "amount")
DATE_HINTS = ("date", "time", "day")
CATEGORY_ALIASES = ("category", "product_category", "segment")
REGION_ALIASES = ("region", "state", "city", "location", "territory")
PRODUCT_ALIASES = ("product", "product_name", "item", "sku", "product_id")
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


def parse_datetime_series(series):
    try:
        return pd.to_datetime(series, errors="coerce", format="mixed")
    except TypeError:
        return pd.to_datetime(series, errors="coerce")


def detect_date_column(df):
    candidates = []
    for column in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[column]):
            return column, pd.to_datetime(df[column], errors="coerce")
        # Accept object or pandas string dtype as text-like for date detection
        if not (pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_string_dtype(df[column])):
            continue
        sample = df[column].dropna().astype(str).head(100)
        if sample.empty:
            continue
        parsed = parse_datetime_series(sample)
        parse_ratio = parsed.notna().mean()
        name_hint = any(token in normalize_name(column) for token in DATE_HINTS)
        if name_hint and parse_ratio >= 0.6 or parse_ratio >= 0.85:
            candidates.append((column, parse_ratio, name_hint))
    if not candidates:
        return None, None
    candidates.sort(key=lambda item: (item[2], item[1]), reverse=True)
    selected = candidates[0][0]
    return selected, parse_datetime_series(df[selected])


def read_csv_if_exists(path):
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    return None if df.empty else df


def read_json_if_exists(path):
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def load_sources(processed_folder, report_folder, model_folder):
    processed_path = Path(processed_folder)
    report_path = Path(report_folder)
    model_path = Path(model_folder)
    return {
        "featured_df": read_csv_if_exists(processed_path / "featured_dataset.csv"),
        "forecast_df": read_csv_if_exists(report_path / "forecast_results.csv"),
        "inventory": read_json_if_exists(report_path / "inventory_report.json"),
        "metrics": read_json_if_exists(model_path / "model_metrics.json"),
        "model_path": model_path / "random_forest_model.pkl",
        "feature_columns_path": model_path / "feature_columns.pkl",
        "paths": {
            "featured": processed_path / "featured_dataset.csv",
            "forecast": report_path / "forecast_results.csv",
            "inventory": report_path / "inventory_report.json",
            "metrics": model_path / "model_metrics.json",
            "exports": report_path / "exports",
        },
    }


def money(value):
    try:
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return "$0.00"


def number(value):
    try:
        return f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return "0.00"


def kpi(label, value, icon, status="Ready"):
    return {"label": label, "value": value, "icon": icon, "status": status}


def build_kpis(sources):
    featured_df = sources["featured_df"]
    forecast_df = sources["forecast_df"]
    inventory = sources["inventory"] or {}
    metrics = sources["metrics"] or {}

    total_revenue = 0
    total_products = 0
    if featured_df is not None:
        revenue_column = find_column(featured_df, REVENUE_ALIASES)
        product_column = find_column(featured_df, PRODUCT_ALIASES)
        if revenue_column:
            total_revenue = pd.to_numeric(featured_df[revenue_column], errors="coerce").fillna(0).sum()
        if product_column:
            total_products = featured_df[product_column].nunique(dropna=True)

    forecast_revenue = 0
    if forecast_df is not None:
        forecast_column = find_column(forecast_df, ("forecast_revenue", "forecast_sales", "forecast", "predicted_revenue"))
        if forecast_column:
            forecast_revenue = pd.to_numeric(forecast_df[forecast_column], errors="coerce").fillna(0).sum()

    low_stock_alerts = 0
    for alert in inventory.get("alerts", []):
        text = f"{alert.get('title', '')} {alert.get('message', '')}".lower()
        if "low" in text or "critical" in text or "reorder" in text:
            low_stock_alerts += 1

    return [
        kpi("Total Revenue", money(total_revenue), "bi-currency-dollar"),
        kpi("Forecast Revenue", money(forecast_revenue), "bi-graph-up-arrow"),
        kpi("Forecast Accuracy (R2)", str(metrics.get("r2_score", "Pending")), "bi-bullseye"),
        kpi("MAE", str(metrics.get("mae", "Pending")), "bi-activity"),
        kpi("RMSE", str(metrics.get("rmse", "Pending")), "bi-speedometer"),
        kpi("Inventory Health Status", inventory.get("classification", "Pending"), "bi-shield-check"),
        kpi("Low Stock Alerts", str(low_stock_alerts), "bi-exclamation-triangle"),
        kpi("Total Products", f"{int(total_products):,}", "bi-box"),
    ]


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


def add_chart(charts, chart_id, title, fig):
    charts.append({"id": chart_id, "title": title, "figure": chart_payload(apply_chart_layout(fig))})


def revenue_series(df):
    if df is None:
        return None, None, None
    date_column, date_values = detect_date_column(df)
    revenue_column = find_column(df, REVENUE_ALIASES)
    if not date_column or revenue_column is None:
        return None, None, None
    data = pd.DataFrame(
        {
            "date": date_values,
            "revenue": pd.to_numeric(df[revenue_column], errors="coerce").fillna(0),
        }
    ).dropna(subset=["date"])
    return data.sort_values("date"), date_column, revenue_column


def build_feature_importance_chart(sources):
    model_path = sources["model_path"]
    feature_columns_path = sources["feature_columns_path"]
    if not model_path.exists() or not feature_columns_path.exists():
        return None
    try:
        model = joblib.load(model_path)
        feature_columns = joblib.load(feature_columns_path)
    except Exception:
        return None
    if not hasattr(model, "feature_importances_"):
        return None
    importance = (
        pd.DataFrame({"feature": feature_columns, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .head(15)
    )
    if importance.empty:
        return None
    fig = px.bar(importance, x="importance", y="feature", orientation="h", title="Feature Importance")
    fig.update_yaxes(categoryorder="total ascending")
    return fig


def build_charts(sources):
    charts = []
    notices = []
    featured_df = sources["featured_df"]
    forecast_df = sources["forecast_df"]
    inventory = sources["inventory"] or {}

    historical, _, revenue_column = revenue_series(featured_df)
    forecast_column = find_column(forecast_df, ("forecast_revenue", "forecast_sales", "forecast", "predicted_revenue")) if forecast_df is not None else None
    forecast_date_column = find_column(forecast_df, ("date", "forecast_date", "day")) if forecast_df is not None else None

    if historical is not None and forecast_df is not None and forecast_column and forecast_date_column:
        forecast_dates = parse_datetime_series(forecast_df[forecast_date_column])
        forecast_plot = pd.DataFrame(
            {
                "date": forecast_dates,
                "revenue": pd.to_numeric(forecast_df[forecast_column], errors="coerce").fillna(0),
                "series": "Forecast",
            }
        ).dropna(subset=["date"])
        historical_plot = historical.tail(180).copy()
        historical_plot["series"] = "Historical"
        combined = pd.concat([historical_plot, forecast_plot], ignore_index=True)
        fig = px.line(combined, x="date", y="revenue", color="series", markers=True, title="Historical Revenue vs Forecast Revenue")
        add_chart(charts, "historicalVsForecastDashboard", "Historical Revenue vs Forecast Revenue", fig)
    else:
        notices.append("Historical vs forecast chart needs featured data and forecast results.")

    if historical is not None:
        monthly = historical.assign(month=historical["date"].dt.to_period("M").astype(str)).groupby("month", as_index=False)["revenue"].sum()
        fig = px.line(monthly, x="month", y="revenue", markers=True, title="Monthly Revenue Trend")
        add_chart(charts, "monthlyRevenueTrendDashboard", "Monthly Revenue Trend", fig)

        category_column = find_column(featured_df, CATEGORY_ALIASES)
        if category_column:
            category = featured_df.copy()
            category["_revenue"] = pd.to_numeric(category[revenue_column], errors="coerce").fillna(0)
            grouped = category.groupby(category_column, as_index=False)["_revenue"].sum().sort_values("_revenue", ascending=False)
            fig = px.bar(grouped, x=category_column, y="_revenue", title="Category Performance")
            add_chart(charts, "categoryPerformanceDashboard", "Category Performance", fig)
        else:
            notices.append("Category performance chart skipped because no category column was found.")

        region_column = find_column(featured_df, REGION_ALIASES)
        if region_column:
            region = featured_df.copy()
            region["_revenue"] = pd.to_numeric(region[revenue_column], errors="coerce").fillna(0)
            grouped = region.groupby(region_column, as_index=False)["_revenue"].sum().sort_values("_revenue", ascending=False)
            fig = px.bar(grouped, x=region_column, y="_revenue", title="Region Performance")
            add_chart(charts, "regionPerformanceDashboard", "Region Performance", fig)
        else:
            notices.append("Region performance chart skipped because no region column was found.")
    else:
        notices.append("Revenue trend charts need processed/featured_dataset.csv with date and revenue columns.")

    if inventory:
        risk = pd.DataFrame(
            {
                "Status": ["Current Stock", "Safety Stock", "Reorder Point", "Recommended Order"],
                "Value": [
                    inventory.get("current_stock", 0),
                    inventory.get("safety_stock", 0),
                    inventory.get("reorder_point", 0),
                    inventory.get("recommended_order_quantity", 0),
                ],
            }
        )
        fig = px.pie(risk, names="Status", values="Value", title="Inventory Health Distribution", hole=0.45)
        add_chart(charts, "inventoryHealthDistributionDashboard", "Inventory Health Distribution", fig)
    else:
        notices.append("Inventory health chart needs reports/inventory_report.json.")

    feature_fig = build_feature_importance_chart(sources)
    if feature_fig is not None:
        add_chart(charts, "featureImportanceDashboard", "Feature Importance", feature_fig)
    else:
        notices.append("Feature importance chart needs a trained model and feature column artifacts.")

    if forecast_df is not None and forecast_column:
        fig = px.histogram(forecast_df, x=forecast_column, nbins=30, title="Forecast Distribution")
        add_chart(charts, "forecastDistributionDashboard", "Forecast Distribution", fig)
    else:
        notices.append("Forecast distribution chart needs reports/forecast_results.csv.")

    return charts, notices


def file_time(path):
    if not path.exists():
        return "Pending"
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")


def latest_export_time(exports_path):
    if not exports_path.exists():
        return "Pending"
    files = [path for path in exports_path.iterdir() if path.is_file()]
    if not files:
        return "Pending"
    latest = max(files, key=lambda path: path.stat().st_mtime)
    return datetime.fromtimestamp(latest.stat().st_mtime).strftime("%Y-%m-%d %H:%M")


def build_widgets(sources):
    paths = sources["paths"]
    return [
        {"label": "Last Upload Time", "value": file_time(paths["featured"]), "icon": "bi-cloud-arrow-up"},
        {"label": "Last Model Training Time", "value": file_time(paths["metrics"]), "icon": "bi-cpu"},
        {"label": "Last Forecast Run", "value": file_time(paths["forecast"]), "icon": "bi-graph-up-arrow"},
        {"label": "Last Report Generated", "value": latest_export_time(paths["exports"]), "icon": "bi-file-earmark-arrow-down"},
    ]


def build_executive_dashboard(processed_folder, report_folder, model_folder):
    sources = load_sources(processed_folder, report_folder, model_folder)
    charts, notices = build_charts(sources)
    return {
        "kpis": build_kpis(sources),
        "charts": charts,
        "widgets": build_widgets(sources),
        "notices": notices,
    }
