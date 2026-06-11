from pathlib import Path

from flask import Blueprint, current_app, flash, redirect, render_template, request, send_from_directory, url_for

from app.utils.reports import ReportExportError, generate_report, report_history

reports_bp = Blueprint("reports", __name__, url_prefix="/reports")


@reports_bp.route("/")
def index():
    report_options = [
        {
            "type": "forecast_pdf",
            "title": "Forecast Report PDF",
            "description": "Forecast KPIs, table, summary, and chart summary.",
            "icon": "bi-file-earmark-pdf",
        },
        {
            "type": "inventory_pdf",
            "title": "Inventory Optimization Report PDF",
            "description": "Stock policy metrics, inventory status, and alerts.",
            "icon": "bi-file-earmark-pdf",
        },
        {
            "type": "executive_pdf",
            "title": "Executive Summary PDF",
            "description": "Revenue outlook, forecast accuracy, inventory health, and recommendations.",
            "icon": "bi-file-earmark-richtext",
        },
        {
            "type": "excel",
            "title": "Excel Report",
            "description": "Workbook with Forecast Data, Inventory Analysis, and Model Metrics sheets.",
            "icon": "bi-file-earmark-spreadsheet",
        },
    ]
    return render_template(
        "reports.html",
        title="Reports",
        report_options=report_options,
        history=report_history(current_app.config["REPORT_FOLDER"]),
    )


@reports_bp.route("/generate", methods=["POST"])
def generate():
    report_type = request.form.get("report_type")
    try:
        output_path = generate_report(
            report_type,
            current_app.config["REPORT_FOLDER"],
            current_app.config["MODEL_FOLDER"],
        )
        flash(f"{output_path.name} generated successfully.", "success")
    except ReportExportError as exc:
        flash(str(exc), "danger")
    except Exception:
        current_app.logger.exception("Report export failed")
        flash("Report export failed because the source files could not be processed.", "danger")
    return redirect(url_for("reports.index"))


@reports_bp.route("/download/<path:filename>")
def download(filename):
    export_dir = Path(current_app.config["REPORT_FOLDER"]) / "exports"
    return send_from_directory(export_dir, filename, as_attachment=True)
