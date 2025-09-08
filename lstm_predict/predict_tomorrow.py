import numpy as np
import torch
from model import StockLSTM
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import tushare as ts
import config
from data_loader import create_sequences

def get_latest_data(ts_code='000001.SZ', days=60):
    try:
        ts.set_token(config.TUSHARE_TOKEN)
        pro = ts.pro_api()
        end_date = datetime.now().strftime('%Y%m%d')  # Tushare uses YYYYMMDD format
        start_date = (datetime.now() - timedelta(days=days*2)).strftime('%Y%m%d')
        
        print(f"Fetching data for {ts_code} from {start_date} to {end_date}")
        
        # Try to get the data
        data = pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        
        if data is None or data.empty:
            print("Warning: No data returned from Tushare API")
            # Try to get the last 100 days as fallback
            data = pro.daily(ts_code=ts_code, limit=100)
            if data is None or data.empty:
                raise ValueError(f"Failed to fetch data for {ts_code}")
        data = data.sort_values('trade_date', ascending=True)  # 按交易日期升序排序
        print(f"Successfully fetched {len(data)} records")
        return data[['open','high','low','close','vol']]
        
    except Exception as e:
        print(f"Error in get_latest_data: {str(e)}")
        print("Please check your Tushare token and internet connection")
        raise

def prepare_input(data, window_size=60, scaler=None):
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data[-window_size:])
    else:
        scaled_data = scaler.transform(data[-window_size:])
    return torch.FloatTensor(scaled_data).unsqueeze(0), scaler

def predict_tomorrow_price(ts_code='000001.SZ'):
    try:
        # 加载模型
        model = StockLSTM()
        model.load_state_dict(torch.load(f'{ts_code}_stock_lstm.pth'))
        model.eval()
        
        # 获取最新数据
        data = get_latest_data(ts_code)
        if data is None or data.empty:
            raise ValueError("No data available for prediction")
            
        print(f"Latest data sample:\n{data.head()}")
        
        # 准备输入数据（使用训练时的scaler）
        _, _, scaler = create_sequences(data)  # 复用训练时的scaler
        input_tensor, _ = prepare_input(data, scaler=scaler)
        
        # 预测
        with torch.no_grad():
            prediction = model(input_tensor).numpy()
        
        # 反归一化
        dummy = np.zeros((1, 5))
        dummy[:, 0] = prediction.flatten()
        predicted_price = scaler.inverse_transform(dummy)[0, 0]
        
        print(f"\n今日收盘价: {data['close'].iloc[-1]:.2f}")
        print(f"预测明日收盘价: {predicted_price:.2f}")
        return predicted_price
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise

if __name__ == '__main__':
    predict_tomorrow_price("000001.SZ")