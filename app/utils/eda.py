import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder


class EDAError(Exception):
    """Raised when exploratory analysis cannot be generated."""


REVENUE_ALIASES = ("revenue", "sales", "total_sales", "sales_amount", "amount")
ORDER_ALIASES = ("order_id", "invoice_id", "transaction_id", "sale_id")
PRODUCT_ALIASES = ("product", "product_name", "item", "sku", "product_id")
CATEGORY_ALIASES = ("category", "product_category", "segment")
REGION_ALIASES = ("region", "state", "city", "location", "territory")
DATE_HINTS = ("date", "time", "day")

PLOTLY_TEMPLATE = "plotly_dark"
CHART_HEIGHT = 390


def load_featured_dataset(processed_folder):
    path = Path(processed_folder) / "featured_dataset.csv"

    if not path.exists():
        raise EDAError("Missing featured dataset. Run Feature Engineering first.")

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise EDAError("The featured dataset is empty.") from exc
    except Exception as exc:
        raise EDAError("The featured dataset could not be read.") from exc

    if df.empty:
        raise EDAError("The featured dataset is empty.")

    return df, path


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
        raise EDAError("Missing date column. Feature Engineering must include a valid date field.")

    candidates.sort(key=lambda item: (item[2], item[1]), reverse=True)
    selected = candidates[0][0]
    return selected, parse_datetime_series(df[selected])


def numeric_column(df, column):
    return pd.to_numeric(df[column], errors="coerce")


def money(value):
    if pd.isna(value):
        return "$0"
    return f"${float(value):,.2f}"


def plain_number(value):
    if pd.isna(value):
        return "0"
    return f"{int(value):,}"


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


def kpi_cards(df, revenue_column, order_column, product_column, category_column, region_column):
    revenue = numeric_column(df, revenue_column).fillna(0)
    total_revenue = revenue.sum()
    total_orders = df[order_column].nunique(dropna=True) if order_column else len(df)
    total_products = df[product_column].nunique(dropna=True) if product_column else 0
    average_revenue = revenue.mean()

    top_category = "Unavailable"
    if category_column:
        category_totals = revenue.groupby(df[category_column]).sum().sort_values(ascending=False)
        if not category_totals.empty:
            top_category = str(category_totals.index[0])

    top_region = "Unavailable"
    if region_column:
        region_totals = revenue.groupby(df[region_column]).sum().sort_values(ascending=False)
        if not region_totals.empty:
            top_region = str(region_totals.index[0])

    return [
        {"label": "Total Revenue", "value": money(total_revenue), "icon": "bi-currency-dollar"},
        {"label": "Total Orders", "value": plain_number(total_orders), "icon": "bi-receipt"},
        {"label": "Total Products", "value": plain_number(total_products), "icon": "bi-box"},
        {"label": "Average Revenue", "value": money(average_revenue), "icon": "bi-graph-up"},
        {"label": "Top Category", "value": top_category, "icon": "bi-tags"},
        {"label": "Top Region", "value": top_region, "icon": "bi-geo-alt"},
    ]


def generate_charts(df, date_column, date_values, revenue_column, product_column, category_column, region_column):
    charts = []
    notices = []
    working = df.copy()
    working[date_column] = date_values
    working["_revenue_value"] = numeric_column(working, revenue_column).fillna(0)
    working = working.dropna(subset=[date_column])

    daily = working.groupby(date_column, as_index=False)["_revenue_value"].sum().sort_values(date_column)
    fig = px.line(daily, x=date_column, y="_revenue_value", markers=True, title="Revenue Trend")
    fig.update_traces(line=dict(width=3))
    fig.update_yaxes(title="Revenue")
    charts.append({"id": "revenueTrend", "title": "Revenue Trend Line Chart", "figure": chart_payload(apply_chart_layout(fig))})

    monthly = (
        working.assign(month_period=working[date_column].dt.to_period("M").astype(str))
        .groupby("month_period", as_index=False)["_revenue_value"]
        .sum()
    )
    fig = px.line(monthly, x="month_period", y="_revenue_value", markers=True, title="Monthly Sales Trend")
    fig.update_yaxes(title="Revenue")
    charts.append({"id": "monthlySalesTrend", "title": "Monthly Sales Trend", "figure": chart_payload(apply_chart_layout(fig))})

    if category_column:
        category = working.groupby(category_column, as_index=False)["_revenue_value"].sum().sort_values("_revenue_value", ascending=False)
        fig = px.bar(category, x=category_column, y="_revenue_value", title="Category Performance")
        fig.update_yaxes(title="Revenue")
        charts.append({"id": "categoryPerformance", "title": "Category Performance Bar Chart", "figure": chart_payload(apply_chart_layout(fig))})
    else:
        notices.append("Category Performance Bar Chart skipped because no category column was found.")

    if region_column:
        region = working.groupby(region_column, as_index=False)["_revenue_value"].sum().sort_values("_revenue_value", ascending=False)
        fig = px.bar(region, x=region_column, y="_revenue_value", title="Region Performance")
        fig.update_yaxes(title="Revenue")
        charts.append({"id": "regionPerformance", "title": "Region Performance Bar Chart", "figure": chart_payload(apply_chart_layout(fig))})
    else:
        notices.append("Region Performance Bar Chart skipped because no region column was found.")

    if product_column:
        products = (
            working.groupby(product_column, as_index=False)["_revenue_value"]
            .sum()
            .sort_values("_revenue_value", ascending=False)
            .head(10)
        )
        fig = px.bar(products, x="_revenue_value", y=product_column, orientation="h", title="Top 10 Products")
        fig.update_yaxes(categoryorder="total ascending")
        fig.update_xaxes(title="Revenue")
        charts.append({"id": "topProducts", "title": "Top 10 Products Chart", "figure": chart_payload(apply_chart_layout(fig))})
    else:
        notices.append("Top 10 Products Chart skipped because no product column was found.")

    fig = px.histogram(working, x="_revenue_value", nbins=30, title="Sales Distribution")
    fig.update_xaxes(title="Revenue")
    fig.update_yaxes(title="Count")
    charts.append({"id": "salesDistribution", "title": "Sales Distribution Histogram", "figure": chart_payload(apply_chart_layout(fig))})

    numeric_df = working.select_dtypes(include="number")
    if numeric_df.shape[1] >= 2:
        corr = numeric_df.corr(numeric_only=True)
        fig = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=corr.columns.tolist(),
                y=corr.columns.tolist(),
                colorscale="Blues",
                zmin=-1,
                zmax=1,
                colorbar=dict(title="Correlation"),
            )
        )
        fig.update_layout(title="Correlation Heatmap")
        charts.append({"id": "correlationHeatmap", "title": "Correlation Heatmap", "figure": chart_payload(apply_chart_layout(fig))})
    else:
        notices.append("Correlation Heatmap skipped because at least two numeric columns are required.")

    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    weekday = (
        working.assign(weekday_name=working[date_column].dt.day_name())
        .groupby("weekday_name", as_index=False)["_revenue_value"]
        .sum()
    )
    weekday["weekday_name"] = pd.Categorical(weekday["weekday_name"], categories=weekday_order, ordered=True)
    weekday = weekday.sort_values("weekday_name")
    fig = px.bar(weekday, x="weekday_name", y="_revenue_value", title="Revenue by Weekday")
    fig.update_yaxes(title="Revenue")
    charts.append({"id": "revenueByWeekday", "title": "Revenue by Weekday", "figure": chart_payload(apply_chart_layout(fig))})

    quarter = (
        working.assign(quarter_name="Q" + working[date_column].dt.quarter.astype(str))
        .groupby("quarter_name", as_index=False)["_revenue_value"]
        .sum()
        .sort_values("quarter_name")
    )
    fig = px.bar(quarter, x="quarter_name", y="_revenue_value", title="Revenue by Quarter")
    fig.update_yaxes(title="Revenue")
    charts.append({"id": "revenueByQuarter", "title": "Revenue by Quarter", "figure": chart_payload(apply_chart_layout(fig))})

    return charts, notices


def build_eda_dashboard(df):
    revenue_column = find_column(df, REVENUE_ALIASES)
    if not revenue_column:
        raise EDAError("Missing revenue column. Expected one of: revenue, sales, total_sales, sales_amount, amount.")

    date_column, date_values = detect_date_column(df)
    valid_dates = date_values.notna().sum()
    if valid_dates == 0:
        raise EDAError("Missing date column. No valid date values were found.")

    order_column = find_column(df, ORDER_ALIASES)
    product_column = find_column(df, PRODUCT_ALIASES)
    category_column = find_column(df, CATEGORY_ALIASES)
    region_column = find_column(df, REGION_ALIASES)

    kpis = kpi_cards(df, revenue_column, order_column, product_column, category_column, region_column)
    charts, notices = generate_charts(
        df,
        date_column,
        date_values,
        revenue_column,
        product_column,
        category_column,
        region_column,
    )

    detected_columns = {
        "Date": str(date_column),
        "Revenue": str(revenue_column),
        "Order": str(order_column) if order_column else "Not found",
        "Product": str(product_column) if product_column else "Not found",
        "Category": str(category_column) if category_column else "Not found",
        "Region": str(region_column) if region_column else "Not found",
    }

    return {
        "kpis": kpis,
        "charts": charts,
        "notices": notices,
        "detected_columns": detected_columns,
        "row_count": int(len(df)),
        "column_count": int(df.shape[1]),
    }
