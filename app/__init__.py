import os
from flask import Flask

from .config import get_config


def create_app(config_class=None):
    app = Flask(__name__)

    if config_class is None:
        config_class = get_config()

    app.config.from_object(config_class)

    # Create required folders
    for folder in (
        app.config["UPLOAD_FOLDER"],
        app.config["PROCESSED_FOLDER"],
        app.config["MODEL_FOLDER"],
        app.config["REPORT_FOLDER"],
        app.config["EXPORT_FOLDER"],
    ):
        os.makedirs(folder, exist_ok=True)

    # Import blueprints
    from .blueprints.dashboard import dashboard_bp
    print("Dashboard blueprint loaded")

    from .blueprints.upload import upload_bp
    print("Upload blueprint loaded")

    from .blueprints.preprocessing import preprocessing_bp
    print("Preprocessing blueprint loaded")

    from .blueprints.feature_engineering import feature_engineering_bp
    print("Feature Engineering blueprint loaded")

    from .blueprints.eda import eda_bp
    print("EDA blueprint loaded")

    from .blueprints.training import training_bp
    print("Training blueprint loaded")

    from .blueprints.forecasting import forecasting_bp
    print("Forecasting blueprint loaded")

    from .blueprints.inventory import inventory_bp
    print("Inventory blueprint loaded")

    from .blueprints.reports import reports_bp
    print("Reports blueprint loaded")

    # Register blueprints
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(upload_bp)
    app.register_blueprint(preprocessing_bp)
    app.register_blueprint(feature_engineering_bp)
    app.register_blueprint(eda_bp)
    app.register_blueprint(training_bp)
    app.register_blueprint(forecasting_bp)
    app.register_blueprint(inventory_bp)
    app.register_blueprint(reports_bp)

    # Test route
    @app.route("/ping")
    def ping():
        return "PONG"

    # Show full traceback in browser
    @app.errorhandler(Exception)
    def handle_exception(e):
        import traceback

        return f"""
        <h1>Application Error</h1>
        <pre>
{traceback.format_exc()}
        </pre>
        """, 500

    return app