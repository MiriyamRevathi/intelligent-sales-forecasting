from pathlib import Path

import pandas as pd


class FeatureEngineeringError(Exception):
    """Raised when feature engineering cannot be completed."""


DATE_HINTS = ("date", "time", "day")
REVENUE_ALIASES = ("revenue", "sales", "total_sales", "sales_amount", "amount")
QUANTITY_ALIASES = ("quantity_sold", "quantity", "qty", "units_sold", "units")
CATEGORY_ALIASES = ("category", "product_category", "segment")
REGION_ALIASES = ("region", "state", "city", "location", "territory")
ORDER_ALIASES = ("order_id", "invoice_id", "transaction_id", "sale_id")


def load_cleaned_dataset(processed_folder):
    path = Path(processed_folder) / "cleaned_dataset.csv"

    if not path.exists():
        raise FeatureEngineeringError("Missing cleaned dataset. Run Data Preprocessing first.")

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise FeatureEngineeringError("The cleaned dataset is empty.") from exc
    except Exception as exc:
        raise FeatureEngineeringError("The cleaned dataset could not be read.") from exc

    if df.empty:
        raise FeatureEngineeringError("The cleaned dataset is empty.")

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
        raise FeatureEngineeringError("Missing date column. Add a valid date column before feature engineering.")

    candidates.sort(key=lambda item: (item[2], item[1]), reverse=True)
    selected = candidates[0][0]
    return selected, parse_datetime_series(df[selected])


def safe_preview_value(value):
    try:
        is_missing = pd.isna(value)
        if not isinstance(is_missing, bool):
            is_missing = False
    except (TypeError, ValueError):
        is_missing = False

    if is_missing:
        return ""

    if hasattr(value, "isoformat"):
        return value.isoformat()

    if isinstance(value, (int, float, str, bool)):
        return value

    return str(value)


def numeric_column(df, column):
    return pd.to_numeric(df[column], errors="coerce")


def sales_value_column(df):
    return find_column(df, REVENUE_ALIASES)


def add_date_features(df, date_column, date_values):
    engineered = df.copy()
    engineered[date_column] = date_values

    features = {
        "year": date_values.dt.year,
        "month": date_values.dt.month,
        "day": date_values.dt.day,
        "weekday": date_values.dt.weekday,
        "quarter": date_values.dt.quarter,
        "week_of_year": date_values.dt.isocalendar().week.astype("Int64"),
        "is_weekend": date_values.dt.weekday.isin([5, 6]).astype(int),
        "month_name": date_values.dt.month_name(),
        "quarter_name": "Q" + date_values.dt.quarter.astype("Int64").astype(str),
    }

    generated = []
    for name, values in features.items():
        engineered[name] = values
        generated.append(name)

    return engineered, generated


def add_business_features(df, date_column):
    engineered = df.copy()
    generated = []
    skipped = []

    revenue_column = find_column(engineered, REVENUE_ALIASES)
    quantity_column = find_column(engineered, QUANTITY_ALIASES)
    category_column = find_column(engineered, CATEGORY_ALIASES)
    region_column = find_column(engineered, REGION_ALIASES)
    order_column = find_column(engineered, ORDER_ALIASES)
    value_column = sales_value_column(engineered)

    if revenue_column and quantity_column:
        revenue = numeric_column(engineered, revenue_column)
        quantity = numeric_column(engineered, quantity_column)
        revenue_per_unit = revenue.divide(quantity.where(quantity != 0))
        engineered["revenue_per_unit"] = revenue_per_unit.fillna(0).astype(float)
        generated.append("revenue_per_unit")
    else:
        skipped.append("revenue_per_unit requires revenue and quantity_sold columns.")

    if revenue_column and order_column:
        revenue = numeric_column(engineered, revenue_column)
        unique_orders = engineered[order_column].nunique(dropna=True)
        engineered["average_order_value"] = revenue.sum() / unique_orders if unique_orders else 0
        generated.append("average_order_value")
    elif revenue_column:
        engineered["average_order_value"] = numeric_column(engineered, revenue_column).mean()
        generated.append("average_order_value")
    else:
        skipped.append("average_order_value requires a revenue or sales column.")

    if value_column:
        values = numeric_column(engineered, value_column).fillna(0)
        if date_column in engineered.columns:
            month_period = engineered[date_column].dt.to_period("M")
            engineered["monthly_sales_total"] = values.groupby(month_period).transform("sum")
            generated.append("monthly_sales_total")
        if category_column:
            engineered["category_sales_total"] = values.groupby(engineered[category_column]).transform("sum")
            generated.append("category_sales_total")
        else:
            skipped.append("category_sales_total requires a category column.")
        if region_column:
            engineered["region_sales_total"] = values.groupby(engineered[region_column]).transform("sum")
            generated.append("region_sales_total")
        else:
            skipped.append("region_sales_total requires a region column.")
    else:
        skipped.append("monthly/category/region sales totals require a revenue or sales column.")

    return engineered, generated, skipped


def feature_statistics(df, generated_features):
    stats = []

    for feature in generated_features:
        series = df[feature]
        if pd.api.types.is_numeric_dtype(series):
            stats.append(
                {
                    "feature": feature,
                    "type": str(series.dtype),
                    "non_null": int(series.notna().sum()),
                    "unique": int(series.nunique(dropna=True)),
                    "mean": round(float(series.mean()), 2) if series.notna().any() else "",
                    "min": safe_preview_value(series.min()) if series.notna().any() else "",
                    "max": safe_preview_value(series.max()) if series.notna().any() else "",
                }
            )
        else:
            stats.append(
                {
                    "feature": feature,
                    "type": str(series.dtype),
                    "non_null": int(series.notna().sum()),
                    "unique": int(series.nunique(dropna=True)),
                    "mean": "",
                    "min": "",
                    "max": "",
                }
            )

    return stats


def feature_preview(df):
    preview_records = []
    for record in df.head(10).to_dict(orient="records"):
        preview_records.append({str(key): safe_preview_value(value) for key, value in record.items()})
    return preview_records


def engineer_features(df):
    original_features = [str(column) for column in df.columns.tolist()]
    date_column, date_values = detect_date_column(df)

    engineered, date_features = add_date_features(df, date_column, date_values)
    engineered, business_features, skipped_features = add_business_features(engineered, date_column)

    generated_features = date_features + business_features
    summary = {
        "date_column": str(date_column),
        "total_original_features": len(original_features),
        "total_generated_features": len(generated_features),
        "original_feature_names": original_features,
        "generated_feature_names": generated_features,
        "skipped_features": skipped_features,
        "all_feature_names": [str(column) for column in engineered.columns.tolist()],
        "preview_records": feature_preview(engineered),
        "feature_statistics": feature_statistics(engineered, generated_features),
        "row_count": int(len(engineered)),
        "column_count": int(engineered.shape[1]),
    }

    return engineered, summary


def save_featured_dataset(df, processed_folder):
    output_path = Path(processed_folder) / "featured_dataset.csv"
    df.to_csv(output_path, index=False)
    return output_path
