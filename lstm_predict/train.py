import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model import StockLSTM
from data_loader import load_stock_data, create_sequences

def train_model(ts_code='000001.SZ', start_date='2020-01-01'):
    data = load_stock_data(ts_code, start_date)
    print(data)
    X, y, scaler = create_sequences(data)
    print(X.shape, y.shape)

    # 转换为PyTorch张量
    X_tensor = torch.FloatTensor(X)
    print(X_tensor.shape)
    y_tensor = torch.FloatTensor(y).unsqueeze(1)
    print(y_tensor.shape)
    
    # 划分训练测试集
    split = int(0.8 * len(X))
    train_data = TensorDataset(X_tensor[:split], y_tensor[:split])
    test_data = TensorDataset(X_tensor[split:], y_tensor[split:])
    print(X_tensor[:split].shape)
    print(y_tensor[:split].shape)
    print(X_tensor[split:].shape)
    print(y_tensor[split:].shape)

    # 初始化模型
    model = StockLSTM()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 训练循环
    for epoch in range(100):
        model.train()
        for inputs, targets in DataLoader(train_data, batch_size=32):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        # 验证
        model.eval()
        with torch.no_grad():
            test_preds = model(X_tensor[split:])
            test_loss = criterion(test_preds, y_tensor[split:])
        
        print(f'Epoch {epoch+1} | Train Loss: {loss.item():.4f} | Test Loss: {test_loss.item():.4f}')
    
    torch.save(model.state_dict(), f'{ts_code}_stock_lstm.pth')
    return model, scaler

if __name__ == '__main__':
    train_model("000001.SZ")
