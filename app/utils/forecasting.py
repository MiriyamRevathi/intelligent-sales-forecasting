import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder


class ForecastingError(Exception):
    """Raised when sales forecasting cannot be completed."""


DATE_HINTS = ("date", "time", "day")
REVENUE_ALIASES = ("revenue", "sales", "total_sales", "sales_amount", "amount")
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


def load_featured_dataset(processed_folder):
    path = Path(processed_folder) / "featured_dataset.csv"
    if not path.exists():
        raise ForecastingError("Missing featured dataset. Run Feature Engineering first.")

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ForecastingError("The featured dataset is empty.") from exc
    except Exception as exc:
        raise ForecastingError("The featured dataset could not be read.") from exc

    if df.empty:
        raise ForecastingError("The featured dataset is empty.")

    return df, path


def load_model_artifacts(model_folder):
    model_path = Path(model_folder) / "random_forest_model.pkl"
    encoders_path = Path(model_folder) / "label_encoders.pkl"
    features_path = Path(model_folder) / "feature_columns.pkl"

    missing = []
    if not model_path.exists():
        missing.append("models/random_forest_model.pkl")
    if not encoders_path.exists():
        missing.append("models/label_encoders.pkl")
    if not features_path.exists():
        missing.append("models/feature_columns.pkl")

    if missing:
        raise ForecastingError(f"Missing model artifact(s): {', '.join(missing)}. Train the model first.")

    try:
        model = joblib.load(model_path)
        label_encoders = joblib.load(encoders_path)
        feature_columns = joblib.load(features_path)
    except Exception as exc:
        raise ForecastingError("Model artifacts could not be loaded.") from exc

    if not feature_columns:
        raise ForecastingError("Missing feature columns. Train the model again.")

    return model, label_encoders, [str(column) for column in feature_columns]


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
        raise ForecastingError("Missing date column. The featured dataset must include a valid date field.")

    candidates.sort(key=lambda item: (item[2], item[1]), reverse=True)
    selected = candidates[0][0]
    return selected, parse_datetime_series(df[selected])


def latest_date(df, date_column, date_values):
    valid_dates = date_values.dropna()
    if valid_dates.empty:
        raise ForecastingError("No valid historical dates were found.")
    return valid_dates.max()


def revenue_column(df):
    column = find_column(df, REVENUE_ALIASES)
    if not column:
        return None
    return column


def generate_future_dates(start_date, horizon_days):
    return pd.date_range(start=start_date + pd.Timedelta(days=1), periods=horizon_days, freq="D")


def add_date_features(future_df, date_column):
    dates = future_df[date_column]
    future_df["year"] = dates.dt.year
    future_df["month"] = dates.dt.month
    future_df["day"] = dates.dt.day
    future_df["weekday"] = dates.dt.weekday
    future_df["quarter"] = dates.dt.quarter
    future_df["week_of_year"] = dates.dt.isocalendar().week.astype("Int64")
    future_df["is_weekend"] = dates.dt.weekday.isin([5, 6]).astype(int)
    future_df["month_name"] = dates.dt.month_name()
    future_df["quarter_name"] = "Q" + dates.dt.quarter.astype("Int64").astype(str)
    return future_df


def historical_defaults(df, date_column, target_column):
    feature_source = df.drop(columns=[target_column], errors="ignore")
    defaults = {}

    for column in feature_source.columns:
        if column == date_column:
            continue

        numeric = pd.to_numeric(feature_source[column], errors="coerce")
        if numeric.notna().mean() >= 0.9:
            value = numeric.tail(30).median()
            defaults[str(column)] = 0 if pd.isna(value) else value
        else:
            mode = feature_source[column].tail(30).astype("string").mode(dropna=True)
            defaults[str(column)] = str(mode.iloc[0]) if not mode.empty else "Unknown"

    return defaults


def generate_future_frame(df, date_column, date_values, horizon_days, target_column):
    start_date = latest_date(df, date_column, date_values)
    future_dates = generate_future_dates(start_date, horizon_days)
    future = pd.DataFrame({date_column: future_dates})
    future = add_date_features(future, date_column)

    defaults = historical_defaults(df, date_column, target_column)
    for column, value in defaults.items():
        if column not in future.columns:
            future[column] = value

    return future, start_date


def encode_unknown_safe(series, encoder):
    known_classes = set(encoder.classes_)
    fallback = encoder.classes_[0] if len(encoder.classes_) else "Unknown"
    clean = series.astype("string").fillna(fallback).astype(str)
    clean = clean.where(clean.isin(known_classes), fallback)
    return encoder.transform(clean)


def align_features(future_df, historical_df, feature_columns, label_encoders):
    aligned = pd.DataFrame(index=future_df.index)

    for column in feature_columns:
        if column in future_df.columns:
            series = future_df[column]
        elif column in historical_df.columns:
            historical = historical_df[column]
            numeric = pd.to_numeric(historical, errors="coerce")
            if numeric.notna().mean() >= 0.9:
                fill_value = numeric.tail(30).median()
                series = pd.Series(0 if pd.isna(fill_value) else fill_value, index=future_df.index)
            else:
                mode = historical.tail(30).astype("string").mode(dropna=True)
                series = pd.Series(str(mode.iloc[0]) if not mode.empty else "Unknown", index=future_df.index)
        else:
            series = pd.Series("Unknown" if column in label_encoders else 0, index=future_df.index)

        if column in label_encoders:
            aligned[column] = encode_unknown_safe(series, label_encoders[column])
        else:
            numeric = pd.to_numeric(series, errors="coerce")
            fill_value = numeric.median()
            aligned[column] = numeric.fillna(0 if pd.isna(fill_value) else fill_value)

    return aligned


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


def safe_value(value):
    if pd.isna(value):
        return ""
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def forecast_table(df):
    table = df.copy()
    table["date"] = table["date"].dt.strftime("%Y-%m-%d")
    table["forecast_revenue"] = table["forecast_revenue"].round(2)
    return table.to_dict(orient="records")


def build_charts(history_df, forecast_df, date_column, target_column):
    charts = []
    historical = history_df[[date_column, target_column]].copy()
    historical[target_column] = pd.to_numeric(historical[target_column], errors="coerce").fillna(0)
    historical = historical.dropna(subset=[date_column]).sort_values(date_column).tail(180)
    historical_plot = historical.rename(columns={date_column: "date", target_column: "revenue"})
    historical_plot["series"] = "Historical"

    forecast_plot = forecast_df[["date", "forecast_revenue"]].rename(columns={"forecast_revenue": "revenue"})
    forecast_plot["series"] = "Forecast"
    combined = pd.concat([historical_plot, forecast_plot], ignore_index=True)

    fig = px.line(combined, x="date", y="revenue", color="series", markers=True, title="Historical vs Forecast")
    fig.update_yaxes(title="Revenue")
    charts.append({"id": "historicalVsForecast", "title": "Historical vs Forecast Line Chart", "figure": chart_payload(apply_chart_layout(fig))})

    fig = px.line(forecast_df, x="date", y="forecast_revenue", markers=True, title="Forecast Trend")
    fig.update_yaxes(title="Forecast Revenue")
    charts.append({"id": "forecastTrend", "title": "Forecast Trend Chart", "figure": chart_payload(apply_chart_layout(fig))})

    fig = px.histogram(forecast_df, x="forecast_revenue", nbins=30, title="Forecast Distribution")
    fig.update_xaxes(title="Forecast Revenue")
    charts.append({"id": "forecastDistribution", "title": "Forecast Distribution Histogram", "figure": chart_payload(apply_chart_layout(fig))})

    return charts


def money(value):
    return f"${float(value):,.2f}"


def percentage(value):
    return f"{float(value):,.2f}%"


def calculate_growth(history_df, date_column, target_column, forecast_total, horizon_days):
    recent = history_df.dropna(subset=[date_column]).sort_values(date_column).tail(horizon_days)
    if recent.empty or not target_column:
        return 0.0

    recent_total = pd.to_numeric(recent[target_column], errors="coerce").fillna(0).sum()
    if recent_total == 0:
        return 0.0

    return ((forecast_total - recent_total) / recent_total) * 100


def save_forecast_results(forecast_df, report_folder):
    output_path = Path(report_folder) / "forecast_results.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    forecast_df.to_csv(output_path, index=False)
    return output_path


def generate_forecast(processed_folder, model_folder, report_folder, horizon_days):
    df, dataset_path = load_featured_dataset(processed_folder)
    model, label_encoders, feature_columns = load_model_artifacts(model_folder)
    date_column, date_values = detect_date_column(df)
    target_column = revenue_column(df)
    if not target_column:
        raise ForecastingError("Missing revenue column. Forecasting requires revenue, sales, total_sales, sales_amount, or amount.")

    future_df, start_date = generate_future_frame(df, date_column, date_values, horizon_days, target_column)
    aligned = align_features(future_df, df, feature_columns, label_encoders)
    predictions = model.predict(aligned)

    forecast_df = pd.DataFrame(
        {
            "date": future_df[date_column],
            "forecast_revenue": predictions,
        }
    )
    forecast_df["forecast_revenue"] = forecast_df["forecast_revenue"].clip(lower=0)
    forecast_df["year"] = future_df["year"]
    forecast_df["month"] = future_df["month"]
    forecast_df["weekday"] = future_df["weekday"]
    forecast_df["quarter"] = future_df["quarter"]

    output_path = save_forecast_results(forecast_df, report_folder)

    total_forecast = float(forecast_df["forecast_revenue"].sum())
    average_daily = float(forecast_df["forecast_revenue"].mean())
    max_row = forecast_df.loc[forecast_df["forecast_revenue"].idxmax()]
    min_row = forecast_df.loc[forecast_df["forecast_revenue"].idxmin()]
    growth = calculate_growth(df.assign(**{date_column: date_values}), date_column, target_column, total_forecast, horizon_days)

    charts = build_charts(df.assign(**{date_column: date_values}), forecast_df, date_column, target_column)

    return {
        "source_filename": dataset_path.name,
        "output_filename": output_path.name,
        "horizon_days": horizon_days,
        "latest_historical_date": start_date.strftime("%Y-%m-%d"),
        "target_column": target_column or "Not found",
        "feature_count": len(feature_columns),
        "kpis": [
            {"label": "Total Forecast Revenue", "value": money(total_forecast), "icon": "bi-currency-dollar"},
            {"label": "Average Daily Forecast", "value": money(average_daily), "icon": "bi-calendar-day"},
            {"label": "Maximum Forecast Day", "value": f"{max_row['date'].strftime('%Y-%m-%d')} - {money(max_row['forecast_revenue'])}", "icon": "bi-arrow-up-circle"},
            {"label": "Minimum Forecast Day", "value": f"{min_row['date'].strftime('%Y-%m-%d')} - {money(min_row['forecast_revenue'])}", "icon": "bi-arrow-down-circle"},
            {"label": "Growth vs Recent Period", "value": percentage(growth), "icon": "bi-graph-up-arrow"},
        ],
        "charts": charts,
        "table": forecast_table(forecast_df.head(100)),
    }
