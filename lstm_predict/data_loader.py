import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tushare as ts
import config

def load_stock_data(ts_code='000001.SZ', start_date='2020-01-01'):
    ts.set_token(config.TUSHARE_TOKEN)
    pro = ts.pro_api()
    data = pro.daily(ts_code=ts_code, start_date=start_date)
    data = data.sort_values('trade_date', ascending=True)  # 按交易日期升序排序
    data.to_csv(f'{ts_code}_stock.csv', index=False)
    return data
    
def create_sequences(data, window_size=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['open','high','low','close','vol']])
    X, y = [], []
    for i in range(len(scaled_data)-window_size):
        X.append(scaled_data[i:i+window_size])
        y.append(scaled_data[i+window_size, 3])  # Close price
    return np.array(X), np.array(y), scaler

if __name__ == '__main__':
    data = load_stock_data()
    X, y, scaler = create_sequences(data)
    print(X.shape, y.shape)