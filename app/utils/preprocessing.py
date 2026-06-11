from pathlib import Path

import pandas as pd


SUPPORTED_EXTENSIONS = {"xlsx", "xls", "json"}


class DatasetLoadError(Exception):
    """Raised when an uploaded dataset cannot be loaded for preprocessing."""


def file_extension(path):
    return Path(path).suffix.lower().lstrip(".")


def latest_uploaded_file(upload_folder, preferred_filename=None):
    upload_path = Path(upload_folder)

    if preferred_filename:
        preferred_path = upload_path / preferred_filename
        if preferred_path.exists() and preferred_path.is_file():
            return preferred_path

    candidates = [
        path
        for path in upload_path.iterdir()
        if path.is_file() and file_extension(path) in SUPPORTED_EXTENSIONS
    ]

    if not candidates:
        raise DatasetLoadError("No uploaded dataset found. Upload an Excel or JSON file first.")

    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_dataset(path):
    extension = file_extension(path)

    try:
        if extension in {"xlsx", "xls"}:
            df = pd.read_excel(path)
        elif extension == "json":
            df = pd.read_json(path)
        else:
            raise DatasetLoadError("Unsupported dataset type. Upload .xlsx, .xls, or .json files.")
    except ValueError as exc:
        raise DatasetLoadError("The uploaded file is empty or has an invalid structure.") from exc
    except Exception as exc:
        raise DatasetLoadError("The uploaded file appears to be corrupted or unreadable.") from exc

    if df.empty:
        raise DatasetLoadError("The uploaded dataset is empty.")

    return df


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


def detect_mixed_types(series):
    type_names = sorted({type(value).__name__ for value in series.dropna()})
    if len(type_names) > 1:
        return f"Mixed values: {', '.join(type_names)}"
    return ""


def parse_datetime_series(series):
    try:
        return pd.to_datetime(series, errors="coerce", format="mixed")
    except TypeError:
        return pd.to_datetime(series, errors="coerce")


def should_convert_to_date(series, column_name):
    if pd.api.types.is_datetime64_any_dtype(series):
        return True

    # Accept object or pandas string dtype as text-like for date detection
    if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
        return False

    sample = series.dropna().astype(str).head(100)
    if sample.empty:
        return False

    parsed = parse_datetime_series(sample)
    parse_ratio = parsed.notna().mean()
    name_hint = any(token in column_name.lower() for token in ["date", "time", "day", "month"])

    return name_hint and parse_ratio >= 0.6 or parse_ratio >= 0.85


def fill_missing_values(df):
    cleaned = df.copy()
    fill_methods = {}
    missing_fixed = 0

    for column in cleaned.columns:
        missing_before = int(cleaned[column].isna().sum())
        if missing_before == 0:
            fill_methods[column] = "None"
            continue

        if pd.api.types.is_numeric_dtype(cleaned[column]):
            fill_value = cleaned[column].median()
            fill_method = "Median"
        else:
            mode_values = cleaned[column].mode(dropna=True)
            fill_value = mode_values.iloc[0] if not mode_values.empty else "Unknown"
            fill_method = "Mode"

        cleaned[column] = cleaned[column].fillna(fill_value)
        missing_fixed += missing_before - int(cleaned[column].isna().sum())
        fill_methods[column] = fill_method

    return cleaned, fill_methods, missing_fixed


def preprocess_dataframe(df):
    original_rows = int(len(df))
    original_missing = df.isna().sum()
    original_dtypes = df.dtypes.astype(str).to_dict()
    invalid_type_issues = {column: detect_mixed_types(df[column]) for column in df.columns}

    duplicate_rows = int(df.duplicated().sum())
    cleaned = df.drop_duplicates().copy()

    date_columns = []
    for column in cleaned.columns:
        if should_convert_to_date(cleaned[column], str(column)):
            converted = parse_datetime_series(cleaned[column])
            if converted.notna().any():
                cleaned[column] = converted
                date_columns.append(str(column))

    cleaned, fill_methods, missing_fixed = fill_missing_values(cleaned)
    final_missing = cleaned.isna().sum()

    data_types = []
    for column in cleaned.columns:
        data_types.append(
            {
                "column": str(column),
                "original_dtype": original_dtypes[column],
                "final_dtype": str(cleaned[column].dtype),
                "missing_before": int(original_missing[column]),
                "missing_after": int(final_missing[column]),
                "missing_fixed": max(int(original_missing[column]) - int(final_missing[column]), 0),
                "fill_method": fill_methods.get(column, "None"),
                "date_detected": str(column) in date_columns,
                "type_issue": invalid_type_issues.get(column) or "Valid",
            }
        )

    preview_records = []
    for record in cleaned.head(10).to_dict(orient="records"):
        preview_records.append({str(key): safe_preview_value(value) for key, value in record.items()})

    summary = {
        "original_rows": original_rows,
        "final_rows": int(len(cleaned)),
        "original_columns": int(df.shape[1]),
        "final_columns": int(cleaned.shape[1]),
        "original_missing_values": int(original_missing.sum()),
        "final_missing_values": int(final_missing.sum()),
        "missing_values_fixed": int(missing_fixed),
        "duplicate_rows_removed": duplicate_rows,
        "date_columns_detected": date_columns,
        "data_types": data_types,
        "column_names": [str(column) for column in cleaned.columns.tolist()],
        "preview_records": preview_records,
    }

    return cleaned, summary


def save_cleaned_dataset(df, processed_folder):
    output_path = Path(processed_folder) / "cleaned_dataset.csv"
    df.to_csv(output_path, index=False)
    return output_path
