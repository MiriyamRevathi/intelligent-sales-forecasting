from pathlib import Path
from uuid import uuid4

import pandas as pd
from flask import Blueprint, current_app, flash, redirect, render_template, request, session, url_for
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename

upload_bp = Blueprint("upload", __name__, url_prefix="/upload")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in current_app.config["ALLOWED_EXTENSIONS"]


def file_extension(filename):
    return filename.rsplit(".", 1)[1].lower()


def readable_size(size_bytes):
    if size_bytes is None:
        return "Unknown"

    units = ["B", "KB", "MB", "GB"]
    size = float(size_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} {unit}"
        size /= 1024


def uploaded_files():
    upload_dir = Path(current_app.config["UPLOAD_FOLDER"])
    return sorted(
        [path.name for path in upload_dir.iterdir() if path.is_file()],
        key=str.lower,
    )


def validate_upload(file):
    if not file or file.filename == "":
        return "Please choose an Excel or JSON file to upload."

    if not allowed_file(file.filename):
        return "Only .xlsx, .xls, and .json files are supported."

    content_length = request.content_length
    max_size = current_app.config["MAX_CONTENT_LENGTH"]
    if content_length and content_length > max_size:
        return f"File is too large. Maximum allowed size is {readable_size(max_size)}."

    return None


def unique_filename(original_filename):
    safe_name = secure_filename(original_filename)
    stem = Path(safe_name).stem
    suffix = Path(safe_name).suffix.lower()
    return f"{stem}_{uuid4().hex[:8]}{suffix}"


def read_dataframe(path, extension):
    if extension in {"xlsx", "xls"}:
        return pd.read_excel(path)

    if extension == "json":
        return pd.read_json(path)

    raise ValueError("Unsupported file extension.")


def session_safe_value(value):
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


def dataframe_metadata(df, filename, file_size):
    preview_records = []
    for record in df.head(10).to_dict(orient="records"):
        preview_records.append({str(key): session_safe_value(value) for key, value in record.items()})

    return {
        "filename": filename,
        "file_size": readable_size(file_size),
        "total_rows": int(df.shape[0]),
        "total_columns": int(df.shape[1]),
        "column_names": [str(column) for column in df.columns.tolist()],
        "preview_records": preview_records,
    }


@upload_bp.app_errorhandler(RequestEntityTooLarge)
def handle_file_too_large(error):
    max_size = current_app.config["MAX_CONTENT_LENGTH"]
    flash(f"File is too large. Maximum allowed size is {readable_size(max_size)}.", "danger")
    return redirect(url_for("upload.index"))


@upload_bp.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("dataset")
        validation_error = validate_upload(file)

        if validation_error:
            flash(validation_error, "danger")
            return redirect(url_for("upload.index"))

        original_filename = secure_filename(file.filename)
        filename = unique_filename(original_filename)
        destination = Path(current_app.config["UPLOAD_FOLDER"]) / filename

        try:
            file.save(destination)
            df = read_dataframe(destination, file_extension(filename))
            metadata = dataframe_metadata(df, filename, destination.stat().st_size)
            session["uploaded_dataframe_metadata"] = metadata
            session.pop("preprocessing_summary", None)
            session.pop("preprocessing_source_filename", None)
            session.pop("cleaned_dataset_filename", None)
            session.pop("feature_engineering_summary", None)
            session.pop("featured_dataset_filename", None)
        except ValueError as exc:
            destination.unlink(missing_ok=True)
            flash(str(exc), "danger")
            return redirect(url_for("upload.index"))
        except Exception:
            destination.unlink(missing_ok=True)
            current_app.logger.exception("Failed to process uploaded dataset")
            flash("The file was uploaded, but it could not be read as a valid dataset.", "danger")
            return redirect(url_for("upload.index"))

        flash(f"{original_filename} uploaded and analyzed successfully.", "success")
        return redirect(url_for("upload.index"))

    return render_template(
        "upload.html",
        title="Data Upload",
        files=uploaded_files(),
        metadata=session.get("uploaded_dataframe_metadata"),
        max_upload_size=readable_size(current_app.config["MAX_CONTENT_LENGTH"]),
    )
