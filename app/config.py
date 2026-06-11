import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key")
    UPLOAD_FOLDER = BASE_DIR / "uploads"
    PROCESSED_FOLDER = BASE_DIR / "processed"
    MODEL_FOLDER = BASE_DIR / "models"
    REPORT_FOLDER = BASE_DIR / "reports"
    EXPORT_FOLDER = REPORT_FOLDER / "exports"
    MAX_CONTENT_LENGTH = 32 * 1024 * 1024
    ALLOWED_EXTENSIONS = {"xlsx", "xls", "json"}


class DevelopmentConfig(Config):
    DEBUG = True
    ENV = "development"


class ProductionConfig(Config):
    DEBUG = False
    ENV = "production"
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"


def get_config():
    environment = os.environ.get("FLASK_ENV", os.environ.get("APP_ENV", "development")).lower()
    if environment == "production":
        return ProductionConfig
    return DevelopmentConfig
