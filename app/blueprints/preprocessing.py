from flask import Blueprint, current_app, flash, redirect, render_template, session, url_for

from app.utils.preprocessing import (
    DatasetLoadError,
    latest_uploaded_file,
    load_dataset,
    preprocess_dataframe,
    save_cleaned_dataset,
)

preprocessing_bp = Blueprint("preprocessing", __name__, url_prefix="/preprocessing")


@preprocessing_bp.route("/", methods=["GET", "POST"])
def index():
    summary = session.get("preprocessing_summary")
    source_filename = session.get("preprocessing_source_filename")
    cleaned_filename = session.get("cleaned_dataset_filename")

    if summary is None:
        try:
            upload_metadata = session.get("uploaded_dataframe_metadata", {})
            source_path = latest_uploaded_file(
                current_app.config["UPLOAD_FOLDER"],
                upload_metadata.get("filename"),
            )
            df = load_dataset(source_path)
            cleaned_df, summary = preprocess_dataframe(df)
            output_path = save_cleaned_dataset(cleaned_df, current_app.config["PROCESSED_FOLDER"])

            source_filename = source_path.name
            cleaned_filename = output_path.name
            session["preprocessing_summary"] = summary
            session["preprocessing_source_filename"] = source_filename
            session["cleaned_dataset_filename"] = cleaned_filename
            session.pop("feature_engineering_summary", None)
            session.pop("featured_dataset_filename", None)
            flash("Dataset preprocessing completed successfully.", "success")
        except DatasetLoadError as exc:
            flash(str(exc), "danger")
            return render_template(
                "preprocessing.html",
                title="Data Preprocessing",
                summary=None,
                source_filename=None,
                cleaned_filename=None,
            )
        except Exception:
            current_app.logger.exception("Preprocessing failed")
            flash("Preprocessing failed because the dataset could not be cleaned.", "danger")
            return render_template(
                "preprocessing.html",
                title="Data Preprocessing",
                summary=None,
                source_filename=None,
                cleaned_filename=None,
            )

    return render_template(
        "preprocessing.html",
        title="Data Preprocessing",
        summary=summary,
        source_filename=source_filename,
        cleaned_filename=cleaned_filename,
    )


@preprocessing_bp.route("/run", methods=["POST"])
def run():
    session.pop("preprocessing_summary", None)
    session.pop("preprocessing_source_filename", None)
    session.pop("cleaned_dataset_filename", None)
    session.pop("feature_engineering_summary", None)
    session.pop("featured_dataset_filename", None)
    return redirect(url_for("preprocessing.index"))
