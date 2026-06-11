import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


class ReportExportError(Exception):
    """Raised when a report cannot be generated."""


EXPORT_FILENAMES = {
    "forecast_pdf": "forecast_report.pdf",
    "inventory_pdf": "inventory_optimization_report.pdf",
    "executive_pdf": "executive_summary.pdf",
    "excel": "intelligent_sales_report.xlsx",
}


def export_folder(report_folder):
    path = Path(report_folder) / "exports"
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_forecast_data(report_folder):
    path = Path(report_folder) / "forecast_results.csv"
    if not path.exists():
        raise ReportExportError("Missing forecast_results.csv. Generate a Sales Forecast first.")

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ReportExportError("Forecast results are empty.") from exc
    except Exception as exc:
        raise ReportExportError("Forecast results could not be read.") from exc

    if df.empty:
        raise ReportExportError("Forecast results are empty.")

    return df, path


def load_inventory_report(report_folder):
    path = Path(report_folder) / "inventory_report.json"
    if not path.exists():
        raise ReportExportError("Missing inventory_report.json. Run Inventory Optimization first.")

    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:
        raise ReportExportError("Inventory report could not be read.") from exc

    if not data:
        raise ReportExportError("Inventory report is empty.")

    return data, path


def load_model_metrics(model_folder):
    path = Path(model_folder) / "model_metrics.json"
    if not path.exists():
        raise ReportExportError("Missing model_metrics.json. Train the model first.")

    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception as exc:
        raise ReportExportError("Model metrics could not be read.") from exc

    if not data:
        raise ReportExportError("Model metrics are empty.")

    return data, path


def value_column(df):
    for column in ("forecast_revenue", "forecast_sales", "forecast", "predicted_revenue", "predicted_sales"):
        if column in df.columns:
            return column
    raise ReportExportError("Forecast results do not contain a forecast revenue column.")


def styles():
    base = getSampleStyleSheet()
    base.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=base["Title"],
            textColor=colors.HexColor("#111827"),
            fontSize=22,
            leading=26,
            spaceAfter=14,
        )
    )
    base.add(
        ParagraphStyle(
            name="SectionTitle",
            parent=base["Heading2"],
            textColor=colors.HexColor("#0f172a"),
            fontSize=14,
            leading=18,
            spaceBefore=12,
            spaceAfter=8,
        )
    )
    base.add(
        ParagraphStyle(
            name="SmallMuted",
            parent=base["Normal"],
            textColor=colors.HexColor("#475569"),
            fontSize=9,
            leading=12,
        )
    )
    return base


def add_header(story, title, source=None):
    s = styles()
    story.append(Paragraph(title, s["ReportTitle"]))
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    detail = f"Generated: {generated}"
    if source:
        detail = f"{detail} | Source: {source}"
    story.append(Paragraph(detail, s["SmallMuted"]))
    story.append(Spacer(1, 0.18 * inch))


def make_table(rows, header=True, col_widths=None):
    table = Table(rows, colWidths=col_widths, repeatRows=1 if header else 0)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a") if header else colors.white),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white if header else colors.HexColor("#111827")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def forecast_summary(df):
    column = value_column(df)
    values = pd.to_numeric(df[column], errors="coerce").fillna(0)
    return {
        "total_forecast_revenue": float(values.sum()),
        "average_daily_forecast": float(values.mean()),
        "maximum_forecast": float(values.max()),
        "minimum_forecast": float(values.min()),
        "forecast_days": int(len(values)),
    }


def currency(value):
    return f"${float(value):,.2f}"


def number(value):
    return f"{float(value):,.2f}"


def forecast_kpi_rows(summary):
    return [
        ["Metric", "Value"],
        ["Total Forecast Revenue", currency(summary["total_forecast_revenue"])],
        ["Average Daily Forecast", currency(summary["average_daily_forecast"])],
        ["Maximum Forecast Day", currency(summary["maximum_forecast"])],
        ["Minimum Forecast Day", currency(summary["minimum_forecast"])],
        ["Forecast Days", str(summary["forecast_days"])],
    ]


def table_preview(df, max_rows=25):
    preview = df.head(max_rows).copy()
    for column in preview.columns:
        preview[column] = preview[column].astype(str)
    return [preview.columns.tolist()] + preview.values.tolist()


def add_forecast_chart_summary(story, df):
    s = styles()
    column = value_column(df)
    values = pd.to_numeric(df[column], errors="coerce").fillna(0)
    buckets = pd.cut(values, bins=min(5, max(1, values.nunique())), duplicates="drop")
    counts = buckets.value_counts().sort_index()
    rows = [["Forecast Range", "Days"]]
    for index, count in counts.items():
        rows.append([str(index), str(int(count))])

    story.append(Paragraph("Generated Chart Summary", s["SectionTitle"]))
    story.append(Paragraph("Forecast distribution summary for the exported forecast period.", s["SmallMuted"]))
    story.append(make_table(rows))


def generate_forecast_pdf(report_folder):
    df, source_path = load_forecast_data(report_folder)
    output_path = export_folder(report_folder) / EXPORT_FILENAMES["forecast_pdf"]
    doc = SimpleDocTemplate(str(output_path), pagesize=landscape(A4), rightMargin=24, leftMargin=24, topMargin=28, bottomMargin=28)
    s = styles()
    story = []

    add_header(story, "Forecast Report", source_path.name)
    summary = forecast_summary(df)
    story.append(Paragraph("Forecast KPIs", s["SectionTitle"]))
    story.append(make_table(forecast_kpi_rows(summary)))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("Forecast Summary", s["SectionTitle"]))
    story.append(Paragraph("This report summarizes predicted revenue across the selected forecast horizon and includes a tabular forecast preview.", s["Normal"]))
    add_forecast_chart_summary(story, df)
    story.append(PageBreak())

    story.append(Paragraph("Forecast Table", s["SectionTitle"]))
    story.append(make_table(table_preview(df, 35)))
    doc.build(story)
    return output_path


def inventory_rows(inventory):
    return [
        ["Metric", "Value"],
        ["Current Stock", number(inventory.get("current_stock", 0))],
        ["Safety Stock", number(inventory.get("safety_stock", 0))],
        ["Reorder Point", number(inventory.get("reorder_point", 0))],
        ["Recommended Order Quantity", number(inventory.get("recommended_order_quantity", 0))],
        ["Inventory Status", str(inventory.get("classification", "Unavailable"))],
        ["Days Until Stockout", number(inventory.get("days_until_stockout", 0))],
    ]


def generate_inventory_pdf(report_folder):
    inventory, source_path = load_inventory_report(report_folder)
    output_path = export_folder(report_folder) / EXPORT_FILENAMES["inventory_pdf"]
    doc = SimpleDocTemplate(str(output_path), pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    s = styles()
    story = []

    add_header(story, "Inventory Optimization Report", source_path.name)
    story.append(Paragraph("Inventory KPIs", s["SectionTitle"]))
    story.append(make_table(inventory_rows(inventory)))
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph("Alerts", s["SectionTitle"]))
    alert_rows = [["Alert", "Message", "Level"]]
    for alert in inventory.get("alerts", []):
        alert_rows.append([alert.get("title", ""), alert.get("message", ""), alert.get("level", "")])
    story.append(make_table(alert_rows))

    story.append(Paragraph("Inventory Recommendation", s["SectionTitle"]))
    recommendation = "Maintain current stock levels."
    if inventory.get("recommended_order_quantity", 0) > 0:
        recommendation = f"Order {number(inventory.get('recommended_order_quantity', 0))} units to cover forecast demand."
    story.append(Paragraph(recommendation, s["Normal"]))
    doc.build(story)
    return output_path


def generate_executive_pdf(report_folder, model_folder):
    forecast_df, forecast_path = load_forecast_data(report_folder)
    inventory, inventory_path = load_inventory_report(report_folder)
    metrics, metrics_path = load_model_metrics(model_folder)

    output_path = export_folder(report_folder) / EXPORT_FILENAMES["executive_pdf"]
    doc = SimpleDocTemplate(str(output_path), pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    s = styles()
    story = []
    summary = forecast_summary(forecast_df)

    add_header(story, "Executive Summary", f"{forecast_path.name}, {inventory_path.name}, {metrics_path.name}")
    rows = [
        ["Executive Metric", "Value"],
        ["Total Revenue", currency(summary["total_forecast_revenue"])],
        ["Forecast Revenue", currency(summary["total_forecast_revenue"])],
        ["Forecast Accuracy", f"R2 {metrics.get('r2_score', 'N/A')} | RMSE {metrics.get('rmse', 'N/A')}"],
        ["Inventory Health", inventory.get("classification", "Unavailable")],
        ["Recommended Order Quantity", number(inventory.get("recommended_order_quantity", 0))],
    ]
    story.append(make_table(rows))

    story.append(Paragraph("Key Recommendations", s["SectionTitle"]))
    recommendations = []
    if inventory.get("recommended_order_quantity", 0) > 0:
        recommendations.append(f"Place a replenishment order for {number(inventory.get('recommended_order_quantity', 0))} units.")
    if inventory.get("classification") in {"Low Stock", "Critical Stock"}:
        recommendations.append("Prioritize stock replenishment and monitor demand during the lead-time window.")
    if inventory.get("classification") == "Overstock":
        recommendations.append("Review purchasing cadence and consider markdowns or stock rebalancing.")
    if not recommendations:
        recommendations.append("Inventory position is healthy; continue monitoring forecast variance and model performance.")

    for item in recommendations:
        story.append(Paragraph(f"- {item}", s["Normal"]))

    doc.build(story)
    return output_path


def flatten_inventory(inventory):
    rows = []
    for key, value in inventory.items():
        if key == "alerts":
            continue
        rows.append({"Metric": key, "Value": value})
    for alert in inventory.get("alerts", []):
        rows.append({"Metric": f"alert_{alert.get('title', '')}", "Value": alert.get("message", "")})
    return pd.DataFrame(rows)


def flatten_metrics(metrics):
    rows = []
    for key, value in metrics.items():
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                rows.append({"Metric": f"{key}.{child_key}", "Value": child_value})
        elif isinstance(value, list):
            rows.append({"Metric": key, "Value": ", ".join(map(str, value))})
        else:
            rows.append({"Metric": key, "Value": value})
    return pd.DataFrame(rows)


def generate_excel_report(report_folder, model_folder):
    forecast_df, _ = load_forecast_data(report_folder)
    inventory, _ = load_inventory_report(report_folder)
    metrics, _ = load_model_metrics(model_folder)

    output_path = export_folder(report_folder) / EXPORT_FILENAMES["excel"]
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        forecast_df.to_excel(writer, sheet_name="Forecast Data", index=False)
        flatten_inventory(inventory).to_excel(writer, sheet_name="Inventory Analysis", index=False)
        flatten_metrics(metrics).to_excel(writer, sheet_name="Model Metrics", index=False)

        for worksheet in writer.book.worksheets:
            worksheet.freeze_panes = "A2"
            for column_cells in worksheet.columns:
                max_length = max(len(str(cell.value)) if cell.value is not None else 0 for cell in column_cells)
                worksheet.column_dimensions[column_cells[0].column_letter].width = min(max(max_length + 2, 12), 48)

    return output_path


def generate_report(report_type, report_folder, model_folder):
    generators = {
        "forecast_pdf": lambda: generate_forecast_pdf(report_folder),
        "inventory_pdf": lambda: generate_inventory_pdf(report_folder),
        "executive_pdf": lambda: generate_executive_pdf(report_folder, model_folder),
        "excel": lambda: generate_excel_report(report_folder, model_folder),
    }
    if report_type not in generators:
        raise ReportExportError("Unknown report type selected.")
    return generators[report_type]()


def report_history(report_folder):
    folder = export_folder(report_folder)
    files = []
    for path in sorted(folder.iterdir(), key=lambda item: item.stat().st_mtime, reverse=True):
        if path.is_file():
            files.append(
                {
                    "filename": path.name,
                    "size_kb": round(path.stat().st_size / 1024, 2),
                    "modified": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
                }
            )
    return files
