from model import StockLSTM
from data_loader import create_sequences
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_model(ts_code):
    model = StockLSTM()
    model.load_state_dict(torch.load(f'{ts_code}_stock_lstm.pth'))
    model.eval()
    return model

def predict_and_plot(ts_code='000001.SZ'):
    # 加载数据
    data = pd.read_csv(f'{ts_code}_stock.csv', index_col='trade_date')
    features = data[['open','high','low','close','vol']]
    
    # 创建相同格式的序列
    X, y, scaler = create_sequences(features)
    X_tensor = torch.FloatTensor(X)
    
    # 划分测试集（与训练时相同比例）
    split = int(0.8 * len(X))
    X_test = X_tensor[split:]
    y_test = y[split:]
    
    # 加载模型预测
    model = load_model(ts_code)
    with torch.no_grad():
        preds = model(X_test).numpy()
    
    # 反归一化处理
    dummy = np.zeros((len(preds), 5))
    dummy[:, 0] = preds.flatten()
    preds = scaler.inverse_transform(dummy)[:, 0]
    
    dummy[:, 0] = y_test.flatten()
    y_test = scaler.inverse_transform(dummy)[:, 0]
    
    # 绘制结果
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='Actual Price', color='blue', alpha=0.6)
    plt.plot(preds, label='Predicted Price', color='red', linestyle='--')
    plt.title(f'{ts_code} Stock Price Prediction')
    plt.xlabel('Trading Days')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True)
    plt.savefig('stock_prediction.png')
    plt.show()

if __name__ == '__main__':
    predict_and_plot(ts_code='000001.SZ')