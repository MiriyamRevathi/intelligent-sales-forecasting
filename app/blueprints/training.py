from flask import Blueprint, current_app, flash, render_template, request

from app.utils.training import TrainingError, dataset_profile, load_featured_dataset, train_random_forest

training_bp = Blueprint("training", __name__, url_prefix="/training")


@training_bp.route("/", methods=["GET", "POST"])
def index():
    result = None
    profile = None
    selected_target = None

    try:
        df, source_path = load_featured_dataset(current_app.config["PROCESSED_FOLDER"])
        profile = dataset_profile(df)
        profile["source_filename"] = source_path.name
    except TrainingError as exc:
        flash(str(exc), "danger")
        return render_template("training.html", title="Model Training", profile=None, result=None)
    except Exception:
        current_app.logger.exception("Unable to load training dataset")
        flash("Training dataset could not be loaded.", "danger")
        return render_template("training.html", title="Model Training", profile=None, result=None)

    if not profile["numeric_columns"]:
        flash("No numeric target column is available for training.", "danger")
        return render_template("training.html", title="Model Training", profile=profile, result=None)

    selected_target = profile["default_target"]

    if request.method == "POST":
        selected_target = request.form.get("target_column") or profile["default_target"]

        try:
            result = train_random_forest(df, selected_target, current_app.config["MODEL_FOLDER"])
            flash("Random Forest model trained successfully.", "success")
        except TrainingError as exc:
            flash(str(exc), "danger")
        except Exception:
            current_app.logger.exception("Model training failed")
            flash("Model training failed because the selected data could not be modeled.", "danger")

    return render_template(
        "training.html",
        title="Model Training",
        profile=profile,
        result=result,
        selected_target=selected_target,
    )
