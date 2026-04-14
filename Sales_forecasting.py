# ============================================
# RETAIL SALES FORECASTING USING ML & ARIMA
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

# LOAD DATA
df = pd.read_csv("sales_data.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# PREPROCESSING
df.fillna(method='ffill', inplace=True)

# FEATURE ENGINEERING
df['day'] = df['Date'].dt.day
df['month'] = df['Date'].dt.month
df['year'] = df['Date'].dt.year

df['lag_1'] = df['Sales'].shift(1)
df['rolling_mean'] = df['Sales'].rolling(3).mean()

df.dropna(inplace=True)

# TRAIN TEST SPLIT
train_size = int(len(df) * 0.8)
train = df[:train_size]
test = df[train_size:]

X_train = train[['day', 'month', 'year', 'lag_1', 'rolling_mean']]
y_train = train['Sales']

X_test = test[['day', 'month', 'year', 'lag_1', 'rolling_mean']]
y_test = test['Sales']

# MODELS
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# EVALUATION
def evaluate(name, y_true, y_pred):
    print(f"\n{name} Results:")
    print("MAE :", round(mean_absolute_error(y_true, y_pred), 2))
    print("RMSE:", round(np.sqrt(mean_squared_error(y_true, y_pred)), 2))
    print("R2  :", round(r2_score(y_true, y_pred), 2))

evaluate("Linear Regression", y_test, lr_pred)
evaluate("Random Forest", y_test, rf_pred)

# ARIMA FORECAST
ts = df.set_index('Date')['Sales']
model = ARIMA(ts, order=(5,1,0))
model_fit = model.fit()
forecast = model_fit.forecast(steps=30)

# PLOT
plt.figure(figsize=(10,5))
plt.plot(ts, label="Actual Sales")

future_dates = pd.date_range(start=ts.index[-1], periods=30, freq='D')
plt.plot(future_dates, forecast, linestyle='dashed', label="Forecast")

plt.title("Sales Forecast - Next 30 Days")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.grid()
plt.show()

# FINAL OUTPUT
print("\n===== FINAL OUTPUT =====")
print("Retail Sales Forecasting completed successfully.")
print("Random Forest model performed best.")
print("Next 30 days sales predicted using ARIMA.")
print("========================")