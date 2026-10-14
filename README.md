# Sales Forecasting Using Time Series Analysis

A portfolio-ready beginner project that explores retail revenue trends, compares forecasting baselines with ARIMA and Holt-Winters, evaluates results on a chronological holdout set, and generates a 30-day forecast.

> **Dataset note:** `data/retail_sales.csv` is synthetic demonstration data generated for this project. It is not real company data. Revenue values are illustrative and have no currency guarantee.

## Features
- Data cleaning and daily time-series aggregation
- Historical trend, monthly seasonality, weekday patterns, and promotion comparison
- Naive and seasonal-naive baselines, ARIMA, and Holt-Winters model comparison
- MAE, RMSE, and safe MAPE evaluation on the final 90 days
- Actual-versus-predicted chart and 30-day forecast CSV/chart
- Business insights report generated from computed results
- Reusable Python pipeline and Jupyter Notebook

## Project structure
```text
Sales-Forecasting-Project/
├── data/retail_sales.csv
├── notebooks/sales_forecasting_analysis.ipynb
├── src/forecasting.py
├── outputs/                  # generated CSVs and charts
├── reports/business_insights_report.md
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

## Setup
Python 3.10 or newer is recommended.

### Windows
```bash
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the complete pipeline
Run from the project root:

```bash
python src/forecasting.py
```

This regenerates the charts and results under `outputs/` and updates the business report under `reports/`.

## Run the notebook
```bash
jupyter notebook
```
Open `notebooks/sales_forecasting_analysis.ipynb` and run the cells from top to bottom. In a hosted notebook environment, upload the whole project folder or open the notebook and ensure the project root is the current working directory.

## Models and evaluation
The last 90 days are held out as test data; earlier dates are used for training. Data is never randomly shuffled. Models are compared using:
- **MAE:** average absolute error in revenue units; lower is better.
- **RMSE:** penalizes larger errors more heavily; lower is better.
- **MAPE:** average absolute percentage error, excluding zero actuals; lower is better.

The pipeline selects the lowest-RMSE model for the final 30-day forecast. The metrics CSV is generated when you run the pipeline; results can vary with library versions.

## Output files
- `outputs/historical_sales_trend.png`
- `outputs/monthly_sales_analysis.png`
- `outputs/weekly_sales_pattern.png`
- `outputs/promotion_comparison.png`
- `outputs/actual_vs_predicted.png`
- `outputs/future_30_day_forecast.png`
- `outputs/model_metrics.csv`
- `outputs/next_30_day_forecast.csv`

Prediction intervals are approximate for models where exact intervals are unavailable. They should not be treated as guaranteed bounds.

## Business use
Forecasts can support inventory planning, staffing, and promotion scheduling. Validate predictions against fresh real data before using them for business decisions. The synthetic data is for learning and portfolio demonstration only.

## Limitations and future improvements
- Replace synthetic data with a licensed real dataset for a real-world study.
- Add external regressors for holidays, discounts, and promotions.
- Test rolling-origin cross-validation and forecast intervals.
- Compare category-level forecasts and incorporate stock availability.

## License
MIT. See `LICENSE`.
