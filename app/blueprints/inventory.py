from flask import Blueprint, current_app, flash, render_template, request

from app.utils.inventory import InventoryOptimizationError, optimize_inventory

inventory_bp = Blueprint("inventory", __name__, url_prefix="/inventory")


@inventory_bp.route("/", methods=["GET", "POST"])
def index():
    result = None
    form_values = {
        "current_stock": request.form.get("current_stock", ""),
        "lead_time_days": request.form.get("lead_time_days", "7"),
    }

    if request.method == "POST":
        try:
            result = optimize_inventory(
                current_app.config["REPORT_FOLDER"],
                form_values["current_stock"],
                form_values["lead_time_days"],
            )
            flash("Inventory optimization completed successfully.", "success")
        except InventoryOptimizationError as exc:
            flash(str(exc), "danger")
        except Exception:
            current_app.logger.exception("Inventory optimization failed")
            flash("Inventory optimization failed because the forecast data could not be analyzed.", "danger")

    return render_template(
        "inventory.html",
        title="Inventory Optimization",
        result=result,
        form_values=form_values,
    )
