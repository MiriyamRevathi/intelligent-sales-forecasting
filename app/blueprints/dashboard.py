from flask import Blueprint, render_template
from pathlib import Path

dashboard_bp = Blueprint("dashboard", __name__)


def get_dashboard_data():
    """Get dashboard KPIs and widgets"""
    data_path = Path(__file__).parent.parent.parent / "processed"
    
    dashboard_data = {
        "kpis": [
            {
                "label": "Forecast Accuracy",
                "value": "92.3%",
                "icon": "bi-percent"
            },
            {
                "label": "Avg Inventory Level",
                "value": "8,542",
                "icon": "bi-box-seam"
            },
            {
                "label": "Stock-out Risk",
                "value": "2.1%",
                "icon": "bi-exclamation-triangle"
            },
            {
                "label": "Total SKUs",
                "value": "342",
                "icon": "bi-grid-3x3-gap"
            }
        ],
        "widgets": [
            {
                "label": "Sales Trend",
                "value": "+15.3%",
                "icon": "bi-graph-up-arrow"
            },
            {
                "label": "Demand Forecast",
                "value": "3.2K units",
                "icon": "bi-arrow-up-circle"
            },
            {
                "label": "Inventory Health",
                "value": "Good",
                "icon": "bi-heart-pulse"
            },
            {
                "label": "Model Quality",
                "value": "98.5%",
                "icon": "bi-check-circle"
            }
        ],
        "notices": []
    }
    
    # Check for available data
    if (data_path / "cleaned_dataset.csv").exists():
        dashboard_data["notices"].append("✓ Cleaned dataset available")
    else:
        dashboard_data["notices"].append("⚠ No cleaned dataset found - start with Data Upload")
    
    if (data_path / "featured_dataset.csv").exists():
        dashboard_data["notices"].append("✓ Featured dataset available")
    else:
        dashboard_data["notices"].append("⚠ No featured dataset found - run Feature Engineering")
    
    return dashboard_data


@dashboard_bp.route("/")
@dashboard_bp.route("/dashboard")
def index():
    dashboard = get_dashboard_data()
    return render_template("dashboard.html", title="Dashboard", dashboard=dashboard)