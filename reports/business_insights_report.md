# Sales Forecasting — Business Insights Report

## Dataset and scope
- Source: synthetic retail dataset created for this portfolio project (not real company data).
- Date range: 2022-01-01 to 2025-12-30.
- Daily observations: 1,460.
- Total recorded revenue: 1,183,793.99 (dataset currency is illustrative).

## Key observations
- Average daily revenue: 810.82.
- Highest-revenue month in the historical data: July 2025.
- Average transaction revenue on promotion days was 4.1% different from non-promotion days. This is an association, not proof that promotions caused the change.
- Best model by held-out RMSE: **ARIMA(2,1,2)**.
- 30-day forecast total: 27,038.09 (model-based estimate, not a guarantee).

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
