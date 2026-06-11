from flask import Blueprint, current_app, flash, render_template, request

from app.utils.forecasting import ForecastingError, generate_forecast

forecasting_bp = Blueprint("forecasting", __name__, url_prefix="/forecasting")


@forecasting_bp.route("/", methods=["GET", "POST"])
def index():
    horizons = [30, 60, 90]
    selected_horizon = 30
    result = None

    if request.method == "POST":
        try:
            selected_horizon = int(request.form.get("horizon_days", 30))
        except ValueError:
            selected_horizon = 30

        if selected_horizon not in horizons:
            flash("Invalid forecast period selected.", "danger")
            selected_horizon = 30
        else:
            try:
                result = generate_forecast(
                    current_app.config["PROCESSED_FOLDER"],
                    current_app.config["MODEL_FOLDER"],
                    current_app.config["REPORT_FOLDER"],
                    selected_horizon,
                )
                flash("Forecast generated successfully.", "success")
            except ForecastingError as exc:
                flash(str(exc), "danger")
            except Exception:
                current_app.logger.exception("Forecast generation failed")
                flash("Forecast generation failed because the model or dataset could not be processed.", "danger")

    return render_template(
        "forecasting.html",
        title="Sales Forecasting",
        horizons=horizons,
        selected_horizon=selected_horizon,
        result=result,
    )
