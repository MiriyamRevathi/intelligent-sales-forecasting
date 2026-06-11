# Intelligent Sales Forecasting System

A Flask-based portfolio project for end-to-end sales forecasting, exploratory analysis, model training, future revenue forecasting, inventory optimization, and executive reporting.

## Project Overview

The Intelligent Sales Forecasting System helps users upload sales datasets, clean and transform them, train a Random Forest regression model, forecast future sales, optimize inventory decisions, and export professional PDF/Excel reports.

The app is designed for portfolio deployment on Render with a Bootstrap 5 dark interface and modular Flask blueprints.

## Features

- Excel and JSON dataset upload
- Data preprocessing with missing-value handling, duplicate removal, date detection, and cleaned CSV output
- Feature engineering for date and business features
- Exploratory Data Analysis with interactive Plotly charts
- Random Forest model training with MAE, RMSE, R2, MAPE, and feature importance
- Sales forecasting for 30, 60, and 90 days
- Inventory optimization with safety stock, reorder point, and risk alerts
- Executive dashboard with operational KPIs and charts
- PDF and Excel report exports

## Screenshots

Add screenshots after deployment:

- Dashboard
- Upload and preprocessing workflow
- EDA charts
- Model training results
- Forecasting page
- Inventory optimization
- Reports export center

## Architecture Diagram

```text
User
  |
  v
Flask App
  |
  +-- Upload Blueprint -> uploads/
  +-- Preprocessing Blueprint -> processed/cleaned_dataset.csv
  +-- Feature Engineering Blueprint -> processed/featured_dataset.csv
  +-- EDA Blueprint -> Plotly dashboard
  +-- Training Blueprint -> models/*.pkl + model_metrics.json
  +-- Forecasting Blueprint -> reports/forecast_results.csv
  +-- Inventory Blueprint -> reports/inventory_report.json
  +-- Reports Blueprint -> reports/exports/
  +-- Dashboard Blueprint -> Executive overview
```

## Tech Stack

- Python 3.10
- Flask
- Bootstrap 5
- Pandas
- NumPy
- Scikit-learn
- Plotly
- OpenPyXL
- Joblib
- ReportLab
- Gunicorn

## Machine Learning Workflow

1. Upload Excel or JSON sales data.
2. Clean data and save `processed/cleaned_dataset.csv`.
3. Generate date and business features into `processed/featured_dataset.csv`.
4. Train a `RandomForestRegressor`.
5. Save model artifacts in `models/`.
6. Generate future forecasts into `reports/forecast_results.csv`.
7. Optimize inventory from forecast demand.
8. Export reports and view executive dashboard.

## Installation Steps

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Local Run Instructions

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000/
```

Optional environment variables:

```bash
SECRET_KEY=your-local-secret
FLASK_ENV=development
```

## Render Deployment Steps

1. Push the project to GitHub.
2. Create a new Render Web Service.
3. Connect the repository.
4. Use the following settings:
   - Runtime: Python
   - Build command: `pip install -r requirements.txt`
   - Start command: Render will use `Procfile`
5. Add environment variables:
   - `SECRET_KEY`: a strong random secret
   - `FLASK_ENV`: `production`
6. Deploy.

The Procfile uses:

```text
web: gunicorn "app:create_app()"
```

This is correct because the project contains an `app/` package with a `create_app()` factory. Using the factory avoids ambiguity between the root `app.py` script and the package name.

## Project Structure

```text
app/
  blueprints/
  static/
  templates/
  utils/
uploads/
processed/
models/
reports/
  exports/
app.py
Procfile
requirements.txt
runtime.txt
README.md
```

## Future Improvements

- Add authentication and role-based access control
- Add CSRF protection
- Move long-running model training and report generation to background jobs
- Store workflow history in a database
- Add cloud object storage for uploaded and generated files
- Add automated test coverage
- Add Docker deployment support
