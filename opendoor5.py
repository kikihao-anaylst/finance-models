"""
Opendoor (OPEN) Weekly Price Forecast - Final Fixed Version
- Fixes Prophet fit TypeError by ensuring ds/y are correct types and no NaNs
- Compatible with XGBoost 1.3.1 (uses xgb.train / DMatrix)
- LSTM + XGBoost + Prophet ensemble, recursive future forecast, plotting & saving
"""

import os
import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import xgboost as xgb

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Prophet (optional)
try:
    from prophet import Prophet
except Exception:
    try:
        from fbprophet import Prophet
    except Exception:
        Prophet = None

# -----------------------------
# User Config
# -----------------------------
TICKER = "OPEN"
START_DATE = "2018-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")
YEARS_TO_PREDICT = 3            # how many years to forecast
SEQ_LENGTH = 12                 # weeks used by LSTM/XGB window
LSTM_EPOCHS = 40
LSTM_BATCH = 16
ENS_WEIGHTS = {"lstm": 0.4, "xgb": 0.4, "prophet": 0.2}
SAVE_DIR = "opendoor_outputs"
os.makedirs(SAVE_DIR, exist_ok=True)

# -----------------------------
# Helper functions
# -----------------------------
def download_price(ticker, start_date, end_date):
    print(f"Downloading {ticker} from {start_date} to {end_date} ...")
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
    df = df[['Open','High','Low','Close','Volume']].dropna()
    return df

def compute_technical_indicators(df):
    df = df.copy()
    df['MA7'] = df['Close'].rolling(7).mean()
    df['MA21'] = df['Close'].rolling(21).mean()
    delta = df['Close'].diff()
    gain = delta.where(delta>0,0)
    loss = -delta.where(delta<0,0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    df['RSI'] = 100 - (100 / (1 + rs))
    ema12 = df['Close'].ewm(span=12,adjust=False).mean()
    ema26 = df['Close'].ewm(span=26,adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['Signal'] = df['MACD'].ewm(span=9,adjust=False).mean()
    df = df.fillna(method='bfill')
    return df

def resample_weekly(df):
    weekly = df.resample('W-FRI').last().ffill()
    weekly.index.name = 'Date'
    return weekly

def create_lstm_sequences(features_np, seq_len):
    X, y = [], []
    for i in range(seq_len, len(features_np)):
        X.append(features_np[i-seq_len:i, :])
        y.append(features_np[i, 0])  # close is column 0 in our feature matrix
    return np.array(X), np.array(y)

def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def evaluate_and_print(y_true, y_pred, label="Model"):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred)/(y_true + 1e-9))) * 100
    print(f"{label} -> MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")
    return mae, rmse, mape

def safe_prepare_prophet_df(series_or_df):
    """
    Accept either a Series (indexed by date) or a DataFrame with Date/Close.
    Returns a cleaned DataFrame with columns ['ds','y'] where ds is datetime and y is numeric (no NaNs).
    """
    if isinstance(series_or_df, pd.Series):
        df = series_or_df.reset_index().rename(columns={'index': 'ds', series_or_df.name: 'y'}) 
    else:
        df = series_or_df.reset_index().rename(columns={series_or_df.columns[0]: 'y', 'index': 'ds'}) if isinstance(series_or_df, pd.DataFrame) else None

    # More robust handling: accept DataFrame with columns Date/Close
    if df is None:
        raise ValueError("Input must be pandas Series or DataFrame with Date index and Close column")
    # Ensure columns named ds/y
    if 'ds' not in df.columns:
        # try to find date-like column
        df.columns = ['ds'] + list(df.columns[1:])
    if 'y' not in df.columns:
        df.rename(columns={df.columns[1]: 'y'}, inplace=True)

    # Convert and coerce types
    df['ds'] = pd.to_datetime(df['ds'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df = df.dropna(subset=['ds','y']).copy()
    df = df.sort_values('ds').reset_index(drop=True)
    return df[['ds','y']]

# -----------------------------
# 1) Data preparation
# -----------------------------
price_daily = download_price(TICKER, START_DATE, END_DATE)
price_daily = compute_technical_indicators(price_daily)

weekly = resample_weekly(price_daily)

feature_cols = ['Close','MA7','MA21','RSI','MACD','Signal','Volume']
weekly = weekly[feature_cols].dropna().copy()
print(f"Prepared weekly data: {weekly.shape[0]} weeks, features: {weekly.columns.tolist()}")

# -----------------------------
# 2) Scaling
# -----------------------------
scaler_all = MinMaxScaler()
scaled_all = scaler_all.fit_transform(weekly.values)  # (n_weeks, n_features)

scaler_close = MinMaxScaler()
scaler_close.fit(weekly[['Close']].values)

# -----------------------------
# 3) Train/Val/Test splits for LSTM
# -----------------------------
X_all, y_all = create_lstm_sequences(scaled_all, SEQ_LENGTH)
n_samples = len(X_all)
train_end = int(n_samples * 0.8)
val_end = train_end + int(n_samples * 0.1)

X_train, X_val, X_test = X_all[:train_end], X_all[train_end:val_end], X_all[val_end:]
y_train, y_val, y_test = y_all[:train_end], y_all[train_end:val_end], y_all[val_end:]

print(f"Samples total={n_samples}, train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

# -----------------------------
# 4) Train LSTM
# -----------------------------
lstm_model = build_lstm_model((SEQ_LENGTH, scaled_all.shape[1]))
lstm_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH, verbose=1)

pred_lstm_scaled = lstm_model.predict(X_test)
pred_lstm = scaler_close.inverse_transform(pred_lstm_scaled.reshape(-1,1))
y_test_actual = scaler_close.inverse_transform(y_test.reshape(-1,1))
lstm_stats = evaluate_and_print(y_test_actual, pred_lstm, "LSTM")
lstm_model.save(os.path.join(SAVE_DIR, "lstm_model.keras"))

# -----------------------------
# 5) XGBoost (use DMatrix + xgb.train for compatibility)
# -----------------------------
def build_flatten_lag_features(scaled_all, seq_len):
    Xf = []
    for i in range(seq_len, len(scaled_all)):
        Xf.append(scaled_all[i-seq_len:i,:].flatten())
    return np.array(Xf)

X_flat = build_flatten_lag_features(scaled_all, SEQ_LENGTH)
y_flat = scaled_all[SEQ_LENGTH:, 0]  # scaled close

# reuse train/val/test indices consistent with LSTM
Xf_train, Xf_val, Xf_test = X_flat[:train_end], X_flat[train_end:val_end], X_flat[val_end:]
yf_train, yf_val, yf_test = y_flat[:train_end], y_flat[train_end:val_end], y_flat[val_end:]

dtrain = xgb.DMatrix(Xf_train, label=yf_train)
dval = xgb.DMatrix(Xf_val, label=yf_val)
dtest = xgb.DMatrix(Xf_test)

params = {"objective":"reg:squarederror", "max_depth":5, "eta":0.03, "verbosity":0}
xgb_model = xgb.train(params, dtrain, num_boost_round=500, evals=[(dval, 'eval')],
                      early_stopping_rounds=20, verbose_eval=False)

pred_xgb_scaled = xgb_model.predict(dtest)
pred_xgb = scaler_close.inverse_transform(pred_xgb_scaled.reshape(-1,1))
xgb_stats = evaluate_and_print(y_test_actual, pred_xgb, "XGBoost")
joblib.dump(xgb_model, os.path.join(SAVE_DIR, "xgb_model.joblib"))

# -----------------------------
# 6) Prophet on weekly Close (robust cleaning)
# -----------------------------
prophet_pred = np.zeros_like(y_test_actual)
prophet_stats = (None, None, None)
if Prophet is not None:
    try:
        prophet_series = weekly['Close']
        prophet_df = safe_prepare_prophet_df(prophet_series)
        if len(prophet_df) < 10:
            print("Not enough data for Prophet after cleaning; skipping Prophet.")
        else:
            m = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=True)
            m.fit(prophet_df)  # cleaned ds/y DataFrame
            # forecast for test period length
            periods_test = len(y_test_actual)
            future_test = m.make_future_dataframe(periods=periods_test, freq='W-FRI')
            forecast_test = m.predict(future_test)
            prophet_pred_vals = forecast_test['yhat'].values[-periods_test:].reshape(-1,1)
            prophet_pred = prophet_pred_vals
            prophet_stats = evaluate_and_print(y_test_actual, prophet_pred, "Prophet")
            joblib.dump(m, os.path.join(SAVE_DIR, "prophet_model.joblib"))
    except Exception as e:
        print("Prophet failed:", e)
        prophet_pred = np.zeros_like(y_test_actual)
        prophet_stats = (None, None, None)
else:
    print("Prophet not installed; skipping Prophet model.")

# -----------------------------
# 7) Ensemble on test set
# -----------------------------
w_l = ENS_WEIGHTS.get('lstm', 0)
w_x = ENS_WEIGHTS.get('xgb', 0)
w_p = ENS_WEIGHTS.get('prophet', 0)
weights_sum = (w_l + w_x + w_p) if (w_l + w_x + w_p) > 0 else 1.0

ens_pred = (w_l*pred_lstm + w_x*pred_xgb + w_p*prophet_pred) / weights_sum
ensemble_stats = evaluate_and_print(y_test_actual, ens_pred, "Ensemble")

# Save test predictions CSV
test_start_idx = len(weekly) - len(y_test_actual)
test_dates = weekly.index[test_start_idx: test_start_idx + len(y_test_actual)]
df_test = pd.DataFrame({
    'date': test_dates,
    'actual': y_test_actual.flatten(),
    'pred_lstm': pred_lstm.flatten(),
    'pred_xgb': pred_xgb.flatten(),
    'pred_prophet': prophet_pred.flatten() if prophet_pred is not None else np.nan,
    'pred_ensemble': ens_pred.flatten()
})
df_test.to_csv(os.path.join(SAVE_DIR, "test_predictions.csv"), index=False)
print("Saved test predictions to CSV.")

# -----------------------------
# 8) Recursive future forecast (LSTM, XGB, Prophet) and ensemble
# -----------------------------
weeks_to_predict = int(52 * YEARS_TO_PREDICT)
print(f"Forecasting {weeks_to_predict} weeks (~{YEARS_TO_PREDICT} years) into the future...")

# prepare last sequences
last_seq = scaled_all[-SEQ_LENGTH:, :].reshape(1, SEQ_LENGTH, scaled_all.shape[1])
last_flat = X_flat[-1].reshape(1, -1)  # ensure 2D

future_preds_lstm = []
future_preds_xgb = []
future_preds_prophet = []

# Prepare Prophet full future if available and safe
prophet_future_vals = np.full((weeks_to_predict,), np.nan)
if Prophet is not None and 'm' in locals():
    try:
        prophet_df_full = safe_prepare_prophet_df(weekly['Close'])
        mp = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=True)
        mp.fit(prophet_df_full)
        future_full = mp.make_future_dataframe(periods=weeks_to_predict, freq='W-FRI')
        forecast_full = mp.predict(future_full)
        prophet_future_vals = forecast_full['yhat'].values[-weeks_to_predict:]
    except Exception as e:
        print("Prophet future failed:", e)
        prophet_future_vals = np.full((weeks_to_predict,), np.nan)

block_size = scaled_all.shape[1]

for i in range(weeks_to_predict):
    # LSTM prediction
    pred_lstm_scaled = lstm_model.predict(last_seq)
    pred_lstm_val = scaler_close.inverse_transform(pred_lstm_scaled.reshape(-1,1))[0,0]
    future_preds_lstm.append(pred_lstm_val)
    # update last_seq: use scaled_close from LSTM scaled prediction
    scaled_close_val = pred_lstm_scaled.reshape(-1,)[0]
    next_row = last_seq[0, -1, :].copy()
    next_row[0] = scaled_close_val
    last_seq = np.append(last_seq[:,1:,:], next_row.reshape(1,1,-1), axis=1)

    # XGBoost prediction
    last_flat_2d = np.array(last_flat).reshape(1, -1)
    dmat = xgb.DMatrix(last_flat_2d)
    pred_xgb_scaled = xgb_model.predict(dmat)[0]
    pred_xgb_val = scaler_close.inverse_transform(np.array([[pred_xgb_scaled]]))[0,0]
    future_preds_xgb.append(pred_xgb_val)
    # update last_flat
    next_flat = last_flat[0, -block_size:].copy()
    next_flat[0] = pred_xgb_scaled
    last_flat = np.roll(last_flat, -block_size)
    last_flat[0, -block_size:] = next_flat

    # Prophet
    future_preds_prophet.append(float(prophet_future_vals[i]) if not np.isnan(prophet_future_vals[i]) else np.nan)

future_preds_lstm = np.array(future_preds_lstm).reshape(-1,1)
future_preds_xgb = np.array(future_preds_xgb).reshape(-1,1)
future_preds_prophet = np.array(future_preds_prophet).reshape(-1,1)

ensemble_future = (w_l*future_preds_lstm + w_x*future_preds_xgb + w_p*future_preds_prophet) / weights_sum

# Build future dates
last_week = weekly.index[-1]
future_index = pd.date_range(start=last_week, periods=weeks_to_predict+1, freq='W-FRI')[1:]

# Save future predictions CSV
df_future = pd.DataFrame({
    'date': future_index,
    'pred_lstm': future_preds_lstm.flatten(),
    'pred_xgb': future_preds_xgb.flatten(),
    'pred_prophet': future_preds_prophet.flatten(),
    'pred_ensemble': ensemble_future.flatten()
})
df_future.to_csv(os.path.join(SAVE_DIR, "future_predictions.csv"), index=False)
print("Saved future predictions to CSV.")

# -----------------------------
# 9) Plot history + future ensemble
# -----------------------------
plt.figure(figsize=(14,6))
plt.plot(weekly.index, weekly['Close'], label='History', color='tab:blue', alpha=0.6)
plt.plot(future_index, ensemble_future, label='Ensemble forecast', color='orange', linewidth=2)
plt.plot(future_index, future_preds_lstm, label='LSTM (future)', linestyle='--', alpha=0.7)
plt.plot(future_index, future_preds_xgb, label='XGB (future)', linestyle=':', alpha=0.7)
if Prophet is not None:
    plt.plot(future_index, future_preds_prophet, label='Prophet (future)', linestyle='-.', alpha=0.7)

upper = ensemble_future.flatten() * 1.10
lower = ensemble_future.flatten() * 0.90
plt.fill_between(future_index, lower, upper, color='orange', alpha=0.2, label='Approx ±10% band')

plt.title(f"{TICKER} Ensemble Forecast - Next {YEARS_TO_PREDICT} Years (Weekly)")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "future_trend.png"), dpi=150)
plt.show()

# -----------------------------
# 10) Save artifacts
# -----------------------------
joblib.dump(scaler_all, os.path.join(SAVE_DIR, "scaler_all.joblib"))
joblib.dump(scaler_close, os.path.join(SAVE_DIR, "scaler_close.joblib"))
pd.DataFrame({'features': weekly.columns}).to_csv(os.path.join(SAVE_DIR, "feature_list.csv"), index=False)

print("\nSummary on test set:")
print("LSTM:", lstm_stats)
print("XGBoost:", xgb_stats)
print("Prophet:", prophet_stats if 'prophet_stats' in locals() else (None,None,None))
print("Ensemble:", ensemble_stats)
print("\nAll artifacts saved in", SAVE_DIR)
print("Done ✅")
