import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# 定義數據
data = {
    "日期": [
        "2023/5/18", "2023/6/16", "2023/7/26", "2023/8/24", "2023/9/26", 
        "2023/10/24", "2023/11/26", "2023/12/12", "2023/12/21", "2023/12/28", 
        "2023/12/30", "2024/1/1"
    ],
    "賴蕭配": [
        0.27, 0.30, 0.33, 0.37, 0.36, 
        0.34, 0.34, 0.36, 0.33, 0.37, 
        0.33, 0.33
    ],
    "侯康配": [
        0.30, 0.23, 0.25, 0.22, 0.26, 
        0.26, 0.31, 0.32, 0.32, 0.33, 
        0.30, 0.30
    ],
    "柯盈配": [
        0.23, 0.33, 0.32, 0.28, 0.28, 
        0.29, 0.23, 0.22, 0.24, 0.22, 
        0.24, 0.22
    ],
    "未決定": [
        0.20, 0.14, 0.10, 0.13, 0.11, 
        0.10, 0.12, 0.09, 0.11, 0.09, 
        0.13, 0.15
    ]
}

# 創建DataFrame
df = pd.DataFrame(data)

# 將日期轉換為日期格式
df['日期'] = pd.to_datetime(df['日期'], format='%Y/%m/%d', errors='coerce')

# 檢查是否有任何日期轉換失敗
if df['日期'].isnull().any():
    print("日期轉換失敗的數據:")
    print(df[df['日期'].isnull()])

# 設置日期為索引
df.set_index('日期', inplace=True)

# 構建ARIMA模型並預測
def forecast_arima(series, steps):
    model = sm.tsa.ARIMA(series, order=(1, 1, 1))  # ARIMA(p,d,q)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast

# 預測到1月13日
forecast_steps = (pd.to_datetime("2024/1/13", format='%Y/%m/%d') - df.index[-1]).days

forecast_results = {}
for column in df.columns:
    forecast_results[column] = forecast_arima(df[column], forecast_steps)

# 整理預測結果
forecast_df = pd.DataFrame(forecast_results)
forecast_df.index = pd.date_range(start=df.index[-1], periods=forecast_steps + 1, freq='D')[1:]

# 確保每一天的預測值相加為1
forecast_df = forecast_df.div(forecast_df.sum(axis=1), axis=0)

# 顯示1月13日的預測結果
print(forecast_df.loc['2024-01-13'])

# 繪製預測結果
plt.figure(figsize=(18, 6))
for i, column in enumerate(df.columns):
    if column == '賴蕭配':
        color = '#7FC26B'
    elif column == '柯盈配':
        color = '#28C7CA'
    elif column == '侯康配':
        color = '#038DD6'
    else:
        color = 'red'
    plt.plot(df.index, df[column], color=color, label=f'Historical {column}')
    plt.plot(forecast_df.index, forecast_df[column], color=color, linestyle='--', label=f'Forecast {column}')

plt.axvline(x=pd.to_datetime("2024/1/13", format='%Y/%m/%d'), color='r', linestyle='--', label='Election Day')

# 添加資料標籤
last_day_forecast = forecast_df.iloc[-1]
for i, (index, value) in enumerate(last_day_forecast.items()):
    plt.text(pd.to_datetime("2024/1/13", format='%Y/%m/%d'), value, f'{index}: {value:.2f}', ha='left', va='center')

plt.legend()
plt.title('Election Result Forecast')
plt.xlabel('Date')
plt.ylabel('Support Percentage')
plt.grid(True)
plt.show()
