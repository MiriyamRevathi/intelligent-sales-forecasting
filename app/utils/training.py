import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class TrainingError(Exception):
    """Raised when model training cannot be completed."""


TARGET_ALIASES = ("revenue", "sales", "total_sales", "sales_amount", "amount")
PLOTLY_TEMPLATE = "plotly_dark"
CHART_HEIGHT = 390


def load_featured_dataset(processed_folder):
    path = Path(processed_folder) / "featured_dataset.csv"

    if not path.exists():
        raise TrainingError("Missing featured dataset. Run Feature Engineering first.")

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise TrainingError("The featured dataset is empty.") from exc
    except Exception as exc:
        raise TrainingError("The featured dataset could not be read.") from exc

    if df.empty:
        raise TrainingError("The featured dataset is empty.")

    return df, path


def normalize_name(name):
    return str(name).strip().lower().replace(" ", "_")


def numeric_target_columns(df):
    numeric_columns = []
    for column in df.columns:
        numeric_series = pd.to_numeric(df[column], errors="coerce")
        if numeric_series.notna().sum() > 0:
            numeric_columns.append(str(column))
    return numeric_columns


def default_target_column(numeric_columns):
    normalized = {normalize_name(column): column for column in numeric_columns}
    for alias in TARGET_ALIASES:
        if alias in normalized:
            return normalized[alias]
    return numeric_columns[0] if numeric_columns else None


def dataset_profile(df):
    numeric_columns = numeric_target_columns(df)
    return {
        "row_count": int(len(df)),
        "column_count": int(df.shape[1]),
        "numeric_columns": numeric_columns,
        "default_target": default_target_column(numeric_columns),
    }


def unsupported_columns(df):
    unsupported = []
    for column in df.columns:
        series = df[column]
        if series.dropna().empty:
            unsupported.append(str(column))
        elif series.apply(lambda value: isinstance(value, (list, dict, set, tuple))).any():
            unsupported.append(str(column))
    return unsupported


def prepare_training_data(df, target_column):
    if target_column not in df.columns:
        raise TrainingError("Invalid target column selected.")

    y = pd.to_numeric(df[target_column], errors="coerce")
    valid_target_mask = y.notna()
    if valid_target_mask.sum() == 0:
        raise TrainingError("The selected target column does not contain valid numeric values.")

    working = df.loc[valid_target_mask].copy()
    y = y.loc[valid_target_mask]

    dropped_columns = unsupported_columns(working)
    x = working.drop(columns=[target_column] + dropped_columns, errors="ignore")

    if x.empty:
        raise TrainingError("No usable feature columns are available after removing unsupported columns.")

    label_encoders = {}
    feature_columns = []
    numeric_features = []
    categorical_features = []

    for column in x.columns:
        series = x[column]

        numeric_series = pd.to_numeric(series, errors="coerce")
        numeric_ratio = numeric_series.notna().mean()

        if numeric_ratio >= 0.9:
            fill_value = numeric_series.median()
            if pd.isna(fill_value):
                fill_value = 0
            x[column] = numeric_series.fillna(fill_value)
            numeric_features.append(str(column))
        else:
            string_series = series.astype("string")
            mode_values = string_series.mode(dropna=True)
            fill_value = mode_values.iloc[0] if not mode_values.empty else "Unknown"
            string_series = string_series.fillna(fill_value).astype(str)
            encoder = LabelEncoder()
            x[column] = encoder.fit_transform(string_series)
            label_encoders[str(column)] = encoder
            categorical_features.append(str(column))

        feature_columns.append(str(column))

    if len(x) < 5:
        raise TrainingError("At least 5 valid rows are required for an 80/20 train/test split.")

    return x, y, {
        "feature_columns": feature_columns,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "dropped_columns": dropped_columns,
        "label_encoders": label_encoders,
    }


def mape_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    non_zero = y_true != 0
    if non_zero.sum() == 0:
        return 0.0
    return float((np.abs(y_true[non_zero] - y_pred[non_zero]) / np.abs(y_true[non_zero])).mean() * 100)


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


def build_training_charts(y_test, predictions, feature_columns, importances):
    results = pd.DataFrame(
        {
            "actual": y_test.reset_index(drop=True),
            "predicted": predictions,
        }
    )
    results["residual"] = results["actual"] - results["predicted"]
    results["absolute_error"] = results["residual"].abs()

    charts = []

    fig = px.scatter(results, x="actual", y="predicted", title="Actual vs Predicted")
    min_value = min(results["actual"].min(), results["predicted"].min())
    max_value = max(results["actual"].max(), results["predicted"].max())
    fig.add_shape(
        type="line",
        x0=min_value,
        y0=min_value,
        x1=max_value,
        y1=max_value,
        line=dict(color="#66d9ef", width=2, dash="dash"),
    )
    fig.update_xaxes(title="Actual")
    fig.update_yaxes(title="Predicted")
    charts.append({"id": "actualVsPredicted", "title": "Actual vs Predicted Scatter Plot", "figure": chart_payload(apply_chart_layout(fig))})

    importance_df = (
        pd.DataFrame({"feature": feature_columns, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(20)
    )
    fig = px.bar(importance_df, x="importance", y="feature", orientation="h", title="Feature Importance")
    fig.update_yaxes(categoryorder="total ascending")
    charts.append({"id": "featureImportance", "title": "Feature Importance Bar Chart", "figure": chart_payload(apply_chart_layout(fig))})

    fig = px.histogram(results, x="residual", nbins=30, title="Residual Distribution")
    fig.update_xaxes(title="Residual")
    charts.append({"id": "residualDistribution", "title": "Residual Distribution Histogram", "figure": chart_payload(apply_chart_layout(fig))})

    fig = px.histogram(results, x="absolute_error", nbins=30, title="Prediction Error Distribution")
    fig.update_xaxes(title="Absolute Error")
    charts.append({"id": "predictionError", "title": "Prediction Error Distribution", "figure": chart_payload(apply_chart_layout(fig))})

    return charts


def save_artifacts(model, label_encoders, feature_columns, metrics, model_folder):
    model_path = Path(model_folder)
    model_path.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, model_path / "random_forest_model.pkl")
    joblib.dump(label_encoders, model_path / "label_encoders.pkl")
    joblib.dump(feature_columns, model_path / "feature_columns.pkl")

    with (model_path / "model_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


def train_random_forest(df, target_column, model_folder):
    x, y, prep = prepare_training_data(df, target_column)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    start = time.perf_counter()
    model.fit(x_train, y_train)
    training_time = time.perf_counter() - start

    predictions = model.predict(x_test)
    rmse = root_mean_squared_error(y_test, predictions)

    metrics = {
        "target_column": str(target_column),
        "mae": round(float(mean_absolute_error(y_test, predictions)), 4),
        "rmse": round(float(rmse), 4),
        "r2_score": round(float(r2_score(y_test, predictions)), 4),
        "mape": round(float(mape_score(y_test, predictions)), 4),
        "training_time_seconds": round(float(training_time), 4),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "feature_count": int(len(prep["feature_columns"])),
        "numeric_feature_count": int(len(prep["numeric_features"])),
        "categorical_feature_count": int(len(prep["categorical_features"])),
        "dropped_columns": prep["dropped_columns"],
        "model_name": "RandomForestRegressor",
        "model_params": {
            "n_estimators": 300,
            "max_depth": 15,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "random_state": 42,
            "n_jobs": -1,
        },
    }

    charts = build_training_charts(y_test, predictions, prep["feature_columns"], model.feature_importances_)
    save_artifacts(model, prep["label_encoders"], prep["feature_columns"], metrics, model_folder)

    return {
        "metrics": metrics,
        "charts": charts,
        "feature_columns": prep["feature_columns"],
        "numeric_features": prep["numeric_features"],
        "categorical_features": prep["categorical_features"],
        "artifact_files": [
            "models/random_forest_model.pkl",
            "models/label_encoders.pkl",
            "models/feature_columns.pkl",
            "models/model_metrics.json",
        ],
    }
