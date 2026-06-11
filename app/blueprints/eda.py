from flask import Blueprint, current_app, flash, render_template

from app.utils.eda import EDAError, build_eda_dashboard, load_featured_dataset

eda_bp = Blueprint("eda", __name__, url_prefix="/eda")


@eda_bp.route("/")
def index():
    try:
        df, source_path = load_featured_dataset(current_app.config["PROCESSED_FOLDER"])
        dashboard = build_eda_dashboard(df)
        dashboard["source_filename"] = source_path.name
    except EDAError as exc:
        flash(str(exc), "danger")
        return render_template("eda.html", title="EDA Analysis", dashboard=None)
    except Exception:
        current_app.logger.exception("EDA dashboard generation failed")
        flash("EDA dashboard generation failed because the featured dataset could not be analyzed.", "danger")
        return render_template("eda.html", title="EDA Analysis", dashboard=None)

    return render_template("eda.html", title="EDA Analysis", dashboard=dashboard)
