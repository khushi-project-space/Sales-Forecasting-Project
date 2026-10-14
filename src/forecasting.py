"""Reusable sales forecasting utilities for the Sales Forecasting portfolio project."""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "retail_sales.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = PROJECT_ROOT / "reports"

def load_and_clean_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Revenue"])
    df = df.drop_duplicates()
    df["Revenue"] = pd.to_numeric(df["Revenue"], errors="coerce")
    df = df.dropna(subset=["Revenue"])
    df = df[df["Revenue"] >= 0].sort_values("Date")
    return df

def make_daily_series(df):
    series = df.groupby("Date")["Revenue"].sum().sort_index()
    full_index = pd.date_range(series.index.min(), series.index.max(), freq="D")
    return series.reindex(full_index).fillna(0).asfreq("D").rename("Revenue")

def safe_mape(y_true, y_pred):
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    mask = actual != 0
    if not mask.any():
        return np.nan
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)

def evaluate(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE_percent": safe_mape(y_true, y_pred),
    }

def train_and_compare(series, test_days=90):
    if len(series) <= test_days + 60:
        raise ValueError("Not enough daily observations for the selected test window.")
    train, test = series.iloc[:-test_days], series.iloc[-test_days:]
    results, predictions = [], {}
    # Naive baseline: each forecast equals the final training observation.
    naive_pred = np.repeat(train.iloc[-1], len(test))
    predictions["Naive"] = pd.Series(naive_pred, index=test.index)
    results.append({"Model": "Naive", **evaluate(test, naive_pred)})
    # Seasonal naive baseline: use the value from the same weekday one week earlier.
    seasonal_values = []
    history = train.tolist()
    for _ in range(len(test)):
        seasonal_values.append(history[-7])
        history.append(seasonal_values[-1])
    predictions["Seasonal Naive (7-day)"] = pd.Series(seasonal_values, index=test.index)
    results.append({"Model": "Seasonal Naive (7-day)", **evaluate(test, seasonal_values)})
    # SARIMA-like seasonal ARIMA can be expensive; a compact ARIMA model is used as a robust default.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ARIMA(train, order=(2, 1, 2), enforce_stationarity=False, enforce_invertibility=False)
            fitted = model.fit()
            arima_pred = np.asarray(fitted.forecast(steps=len(test)))
        predictions["ARIMA(2,1,2)"] = pd.Series(arima_pred, index=test.index)
        results.append({"Model": "ARIMA(2,1,2)", **evaluate(test, arima_pred)})
    except Exception as exc:
        print(f"ARIMA could not be fitted: {exc}")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=7,
                                       initialization_method="estimated").fit(optimized=True)
            ets_pred = np.asarray(fit.forecast(len(test)))
        predictions["Holt-Winters (weekly)"] = pd.Series(ets_pred, index=test.index)
        results.append({"Model": "Holt-Winters (weekly)", **evaluate(test, ets_pred)})
    except Exception as exc:
        print(f"Holt-Winters could not be fitted: {exc}")
    return train, test, pd.DataFrame(results).sort_values("RMSE"), predictions

def save_eda_charts(df, daily):
    OUTPUT_DIR.mkdir(exist_ok=True)
    plt.figure(figsize=(12, 5))
    daily.plot()
    plt.title("Historical Daily Revenue")
    plt.xlabel("Date"); plt.ylabel("Revenue"); plt.tight_layout()
    plt.savefig(OUTPUT_DIR/"historical_sales_trend.png", dpi=160); plt.close()

    monthly = daily.resample("MS").sum()
    plt.figure(figsize=(12, 5))
    monthly.plot(kind="bar")
    plt.title("Monthly Revenue")
    plt.xlabel("Month"); plt.ylabel("Revenue")
    plt.xticks(rotation=60); plt.tight_layout()
    plt.savefig(OUTPUT_DIR/"monthly_sales_analysis.png", dpi=160); plt.close()

    weekday = df.groupby("Day_Of_Week")["Revenue"].mean().reindex(
        ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
    plt.figure(figsize=(10, 5))
    weekday.plot(kind="bar")
    plt.title("Average Revenue by Day of Week")
    plt.xlabel("Day"); plt.ylabel("Average revenue")
    plt.xticks(rotation=35); plt.tight_layout()
    plt.savefig(OUTPUT_DIR/"weekly_sales_pattern.png", dpi=160); plt.close()

    promo = df.groupby("Promotion")["Revenue"].mean().reindex([0,1])
    promo.index = ["No Promotion", "Promotion"]
    plt.figure(figsize=(8, 5))
    promo.plot(kind="bar")
    plt.title("Average Transaction Revenue: Promotion Comparison")
    plt.ylabel("Average revenue"); plt.xticks(rotation=0); plt.tight_layout()
    plt.savefig(OUTPUT_DIR/"promotion_comparison.png", dpi=160); plt.close()

def run_pipeline():
    OUTPUT_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    df = load_and_clean_data()
    daily = make_daily_series(df)
    save_eda_charts(df, daily)
    train, test, metrics, predictions = train_and_compare(daily)
    metrics.to_csv(OUTPUT_DIR/"model_metrics.csv", index=False)
    best_name = metrics.iloc[0]["Model"]
    best_pred = predictions[best_name]
    plt.figure(figsize=(12, 5))
    train.tail(180).plot(label="Training history")
    test.plot(label="Actual", linewidth=2)
    best_pred.plot(label=f"Predicted — {best_name}", linestyle="--")
    plt.title("Actual vs Predicted Revenue (Test Period)")
    plt.xlabel("Date"); plt.ylabel("Revenue"); plt.legend(); plt.tight_layout()
    plt.savefig(OUTPUT_DIR/"actual_vs_predicted.png", dpi=160); plt.close()

    # Refit the selected model on all observed data for a 30-day operational forecast.
    future_dates = pd.date_range(daily.index.max() + pd.Timedelta(days=1), periods=30, freq="D")
    if best_name == "ARIMA(2,1,2)":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            final_fit = ARIMA(daily, order=(2,1,2), enforce_stationarity=False,
                              enforce_invertibility=False).fit()
            fc_obj = final_fit.get_forecast(steps=30)
            forecast = np.asarray(fc_obj.predicted_mean)
            try:
                ci = fc_obj.conf_int()
                lower, upper = np.asarray(ci.iloc[:,0]), np.asarray(ci.iloc[:,1])
            except Exception:
                lower, upper = forecast*0.9, forecast*1.1
    elif best_name == "Holt-Winters (weekly)":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            final_fit = ExponentialSmoothing(daily, trend="add", seasonal="add", seasonal_periods=7,
                                             initialization_method="estimated").fit(optimized=True)
            forecast = np.asarray(final_fit.forecast(30))
        spread = float(np.std(daily.diff().dropna()))
        lower, upper = forecast - 1.96*spread, forecast + 1.96*spread
    elif best_name == "Seasonal Naive (7-day)":
        forecast = np.array([daily.iloc[-7 + (i % 7)] for i in range(30)])
        spread = float(np.std(daily.diff().dropna()))
        lower, upper = forecast - 1.96*spread, forecast + 1.96*spread
    else:
        forecast = np.repeat(daily.iloc[-1], 30)
        spread = float(np.std(daily.diff().dropna()))
        lower, upper = forecast - 1.96*spread, forecast + 1.96*spread
    forecast = np.maximum(forecast, 0)
    lower, upper = np.maximum(lower, 0), np.maximum(upper, 0)
    future = pd.DataFrame({"Date": future_dates, "Predicted_Revenue": forecast,
                           "Lower_95_Approx": lower, "Upper_95_Approx": upper})
    future.to_csv(OUTPUT_DIR/"next_30_day_forecast.csv", index=False)
    plt.figure(figsize=(12, 5))
    daily.tail(120).plot(label="Historical revenue")
    future.set_index("Date")["Predicted_Revenue"].plot(label="30-day forecast", linestyle="--")
    plt.fill_between(future_dates, lower, upper, alpha=0.2, label="Approx. prediction interval")
    plt.title(f"Next 30 Days Revenue Forecast — {best_name}")
    plt.xlabel("Date"); plt.ylabel("Revenue"); plt.legend(); plt.tight_layout()
    plt.savefig(OUTPUT_DIR/"future_30_day_forecast.png", dpi=160); plt.close()

    # Automatically generated, data-grounded report.
    overall_start, overall_end = daily.index.min().date(), daily.index.max().date()
    total_revenue = float(df["Revenue"].sum())
    promo_avg = df.groupby("Promotion")["Revenue"].mean()
    promo_delta = ((promo_avg.get(1, np.nan) / promo_avg.get(0, np.nan)) - 1) * 100
    best_month = daily.resample("MS").sum().idxmax().strftime("%B %Y")
    avg_daily = float(daily.mean())
    forecast_total = float(future["Predicted_Revenue"].sum())
    report = f"""# Sales Forecasting — Business Insights Report

## Dataset and scope
- Source: synthetic retail dataset created for this portfolio project (not real company data).
- Date range: {overall_start} to {overall_end}.
- Daily observations: {len(daily):,}.
- Total recorded revenue: {total_revenue:,.2f} (dataset currency is illustrative).

## Key observations
- Average daily revenue: {avg_daily:,.2f}.
- Highest-revenue month in the historical data: {best_month}.
- Average transaction revenue on promotion days was {promo_delta:.1f}% different from non-promotion days. This is an association, not proof that promotions caused the change.
- Best model by held-out RMSE: **{best_name}**.
- 30-day forecast total: {forecast_total:,.2f} (model-based estimate, not a guarantee).

## Model evaluation
See `../outputs/model_metrics.csv` for the actual test-set MAE, RMSE, and MAPE results. The test set is the last 90 days and the split preserves time order.

## Business recommendations
1. Use the 30-day forecast as a planning signal, and refresh it when new actual sales arrive.
2. Compare predicted demand with inventory and supplier lead times before placing replenishment orders.
3. Evaluate promotions using margin and units sold as well as revenue; do not infer causality from a simple comparison.
4. Track errors weekly. If forecast error increases during holidays or promotions, add more event-specific features and test a suitable exogenous-variable model.
5. Treat prediction intervals as approximate uncertainty ranges, not guaranteed bounds.

## Limitations
- Data is synthetic and designed for demonstration; it does not represent a real retailer.
- Promotion effects are not causal estimates.
- The base time-series models primarily use historical revenue; they do not explicitly model all product-level, holiday, discount, or promotion variables.
- A 30-day forecast becomes less reliable when market conditions change abruptly.
"""
    (REPORT_DIR/"business_insights_report.md").write_text(report, encoding="utf-8")
    print(f"Pipeline complete. Best model: {best_name}")
    print(metrics.to_string(index=False))
    return metrics, future

if __name__ == "__main__":
    run_pipeline()
