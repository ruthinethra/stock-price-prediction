# Stock Price Prediction

Predicts the next trading day's closing price from technical indicators and compares several ML models.

## Quick start
```bash
pip install -r requirements.txt

# Option A: download from Yahoo Finance
python stock_price_prediction.py --ticker AAPL --start 2015-01-01

# Option B: Kaggle / local CSV (columns: Date, Open, High, Low, Close, Volume)
python stock_price_prediction.py --csv path/to/data.csv

# Add the LSTM (needs: pip install tensorflow); skip with --no_lstm
```
Kaggle datasets to try: "Huge Stock Market Dataset", "S&P 500 stock data", "Tesla Stock Price".

## Pipeline
1. **Data**: Yahoo Finance or Kaggle CSV, sorted by date.
2. **Features**: lagged closes, SMA (5/10/20/50), EMA, MACD, RSI-14, Bollinger width, volatility, high-low range, volume change.
3. **Target**: next-day return, converted back to a price.
4. **Split**: chronological 80/20 (no shuffling); scaler fit on train only to avoid leakage.
5. **Models**: Naive baseline, Ridge, Random Forest, Gradient Boosting, optional LSTM.
6. **Metrics**: RMSE, MAE, MAPE, R², directional accuracy.
7. **Outputs** (`outputs/`): `metrics.csv`, `price_history.png`, `predictions.png`, `feature_importance.png`.

## Notes for your report
- R² near 0.99 is mostly because tomorrow's price is close to today's; always compare against the naive baseline and check directional accuracy.
- Ideas to extend: walk-forward cross-validation, news sentiment features, hyperparameter tuning, multi-day forecasts, a Streamlit dashboard.

*Educational project, not financial advice.*
