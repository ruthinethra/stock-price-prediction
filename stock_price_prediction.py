"""
Stock Price Prediction
----------------------
Predicts the NEXT trading day's closing price (via predicted return) using
technical indicators.

Data sources (pick one):
  1. Yahoo Finance (auto-download):  python stock_price_prediction.py --ticker AAPL
  2. Kaggle / local CSV:             python stock_price_prediction.py --csv data.csv

CSV needs columns: Date, Open, High, Low, Close, Volume
(e.g. Kaggle "Huge Stock Market Dataset" or "S&P 500 stock data").

Models compared: Naive baseline, Linear Regression (Ridge), Random Forest,
Gradient Boosting, and an optional LSTM (if TensorFlow is installed).
"""
import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")  # works without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------- 1. DATA
def load_data(ticker, csv_path, start, end):
    if csv_path:
        df = pd.read_csv(csv_path)
        df.columns = [c.strip().title() for c in df.columns]
    else:
        import yfinance as yf
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").drop_duplicates("Date").set_index("Date")
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    print(f"Loaded {len(df)} rows: {df.index.min().date()} -> {df.index.max().date()}")
    return df


# ------------------------------------------------------ 2. FEATURE ENGINEERING
def rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    return 100 - 100 / (1 + gain / (loss + 1e-9))


def add_features(df):
    d = df.copy()
    c = d["Close"]
    d["Return_1d"] = c.pct_change()
    for lag in (1, 2, 3, 5, 10):
        d[f"Close_lag{lag}"] = c.shift(lag)
    for w in (5, 10, 20, 50):
        d[f"SMA_{w}"] = c.rolling(w).mean()
    d["EMA_12"] = c.ewm(span=12, adjust=False).mean()
    d["EMA_26"] = c.ewm(span=26, adjust=False).mean()
    d["MACD"] = d["EMA_12"] - d["EMA_26"]
    d["RSI_14"] = rsi(c)
    d["Volatility_10"] = d["Return_1d"].rolling(10).std()
    d["BB_width"] = (c.rolling(20).std() * 4) / c.rolling(20).mean()
    d["HL_range"] = (d["High"] - d["Low"]) / c
    d["Vol_change"] = d["Volume"].pct_change()
    # Target: next day's RETURN (shift -1 => no look-ahead leakage in features).
    # Predicting returns (stationary) works far better for tree models than raw
    # prices, which trend outside the range seen in training.
    d["Target"] = c.shift(-1) / c - 1
    return d.replace([np.inf, -np.inf], np.nan).dropna()


# ------------------------------------------------------------- 3. METRICS
def evaluate(name, y_true, y_pred, prev_close):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    # Directional accuracy: did we call up/down correctly vs. today's close?
    dir_acc = np.mean(np.sign(y_pred - prev_close) == np.sign(y_true - prev_close)) * 100
    return {"Model": name, "RMSE": rmse, "MAE": mae, "MAPE %": mape,
            "R2": r2, "Direction Acc %": dir_acc}


# -------------------------------------------------------------- 4. LSTM
def run_lstm(X_tr, y_tr, X_te, seq_len=30, epochs=30):
    try:
        from tensorflow.keras.callbacks import EarlyStopping
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.models import Sequential
    except ImportError:
        print("TensorFlow not installed -> skipping LSTM.")
        return None

    # training sequences
    Xs_tr = np.array([X_tr[i - seq_len:i] for i in range(seq_len, len(X_tr) + 1)])
    ys_tr = y_tr[seq_len - 1:]
    # test sequences use the tail of train as context
    full = np.vstack([X_tr[-(seq_len - 1):], X_te])
    Xs_te = np.array([full[i:i + seq_len] for i in range(len(X_te))])

    model = Sequential([
        LSTM(64, input_shape=(seq_len, X_tr.shape[1])),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(Xs_tr, ys_tr, epochs=epochs, batch_size=32, validation_split=0.1,
              callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
              verbose=0)
    return model.predict(Xs_te, verbose=0).ravel()


# ---------------------------------------------------------------- 5. MAIN
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="AAPL")
    ap.add_argument("--csv", default=None, help="Path to Kaggle/local CSV")
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--no_lstm", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    raw = load_data(args.ticker, args.csv, args.start, args.end)
    data = add_features(raw)
    feature_cols = [c for c in data.columns if c != "Target"]

    # Chronological split (never shuffle time series!)
    split = int(len(data) * (1 - args.test_size))
    train, test = data.iloc[:split], data.iloc[split:]
    scaler = StandardScaler().fit(train[feature_cols])  # fit on train only
    X_tr, X_te = scaler.transform(train[feature_cols]), scaler.transform(test[feature_cols])
    y_tr = train["Target"].values                      # next-day returns
    prev_close = test["Close"].values
    y_te = prev_close * (1 + test["Target"].values)    # actual next-day close

    models = {
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=12,
                                               n_jobs=-1, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, max_depth=3,
                                                       learning_rate=0.05, random_state=42),
    }

    results, preds = [], {}
    # Baseline: tomorrow's close = today's close
    preds["Naive (yesterday)"] = prev_close
    results.append(evaluate("Naive (yesterday)", y_te, prev_close, prev_close))

    for name, m in models.items():
        m.fit(X_tr, y_tr)
        preds[name] = prev_close * (1 + m.predict(X_te))
        results.append(evaluate(name, y_te, preds[name], prev_close))

    if not args.no_lstm:
        p = run_lstm(X_tr, y_tr, X_te)
        if p is not None:
            p = prev_close * (1 + p)
            preds["LSTM"] = p
            results.append(evaluate("LSTM", y_te, p, prev_close))

    res = pd.DataFrame(results).set_index("Model").round(3)
    print("\n=== Test-set results ===")
    print(res.to_string())
    res.to_csv(f"{args.out}/metrics.csv")

    # ---- Plot 1: price history + split
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(raw.index, raw["Close"], lw=1)
    ax.axvline(test.index[0], color="red", ls="--", label="Train | Test")
    ax.set_title("Closing Price History"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{args.out}/price_history.png", dpi=130); plt.close(fig)

    # ---- Plot 2: actual vs predicted
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(test.index, y_te, label="Actual", color="black", lw=1.5)
    for name, p in preds.items():
        if name.startswith("Naive"):
            continue
        ax.plot(test.index[-len(p):], p, label=name, lw=1, alpha=.8)
    ax.set_title("Actual vs Predicted Next-Day Close (Test Set)")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{args.out}/predictions.png", dpi=130); plt.close(fig)

    # ---- Plot 3: feature importance (Random Forest)
    rf = models["Random Forest"]
    imp = pd.Series(rf.feature_importances_, index=feature_cols).nlargest(12)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    imp[::-1].plot.barh(ax=ax); ax.set_title("Top Features (Random Forest)")
    fig.tight_layout(); fig.savefig(f"{args.out}/feature_importance.png", dpi=130); plt.close(fig)

    # ---- Forecast for the next (unseen) trading day using the best model
    best = res.drop(index="Naive (yesterday)")["RMSE"].idxmin()
    if best in models:
        latest = scaler.transform(data[feature_cols].iloc[[-1]])
        nxt = data["Close"].iloc[-1] * (1 + models[best].predict(latest)[0])
        print(f"\nBest model: {best}")
        print(f"Last close: {data['Close'].iloc[-1]:.2f}  ->  "
              f"predicted next close: {nxt:.2f}")
    print(f"\nCharts and metrics saved in ./{args.out}/")
    print("Disclaimer: educational project, not financial advice.")


if __name__ == "__main__":
    main()
