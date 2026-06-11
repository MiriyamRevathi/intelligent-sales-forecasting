from flask import Blueprint, current_app, flash, redirect, render_template, session, url_for

from app.utils.feature_engineering import (
    FeatureEngineeringError,
    engineer_features,
    load_cleaned_dataset,
    save_featured_dataset,
)

feature_engineering_bp = Blueprint("feature_engineering", __name__, url_prefix="/feature-engineering")


@feature_engineering_bp.route("/")
def index():
    summary = session.get("feature_engineering_summary")
    output_filename = session.get("featured_dataset_filename")

    if summary is None:
        try:
            df, source_path = load_cleaned_dataset(current_app.config["PROCESSED_FOLDER"])
            featured_df, summary = engineer_features(df)
            output_path = save_featured_dataset(featured_df, current_app.config["PROCESSED_FOLDER"])

            summary["source_filename"] = source_path.name
            output_filename = output_path.name
            session["feature_engineering_summary"] = summary
            session["featured_dataset_filename"] = output_filename
            flash("Feature engineering completed successfully.", "success")
        except FeatureEngineeringError as exc:
            flash(str(exc), "danger")
            return render_template(
                "feature_engineering.html",
                title="Feature Engineering",
                summary=None,
                output_filename=None,
            )
        except Exception:
            current_app.logger.exception("Feature engineering failed")
            flash("Feature engineering failed because the dataset could not be transformed.", "danger")
            return render_template(
                "feature_engineering.html",
                title="Feature Engineering",
                summary=None,
                output_filename=None,
            )

    return render_template(
        "feature_engineering.html",
        title="Feature Engineering",
        summary=summary,
        output_filename=output_filename,
    )


@feature_engineering_bp.route("/run", methods=["POST"])
def run():
    session.pop("feature_engineering_summary", None)
    session.pop("featured_dataset_filename", None)
    return redirect(url_for("feature_engineering.index"))
