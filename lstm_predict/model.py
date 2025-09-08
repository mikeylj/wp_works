import torch.nn as nn

class StockLSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                          batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

class StockLSTM_BID(nn.Module):
    def __init__(self, input_size=5, hidden_size=128, num_layers=2, dropout=0.2):
        """
        Bidirectional Multi-layer LSTM for stock price prediction
        
        Args:
            input_size (int): Number of input features (default: 5 for OHLCV)
            hidden_size (int): Number of features in the hidden state (default: 128)
            num_layers (int): Number of recurrent layers (default: 2)
            dropout (float): Dropout value (default: 0.2)
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True  # Enable bidirectional LSTM
        )
        
        # Since the LSTM is bidirectional, the output size is hidden_size * 2
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # Bidirectional doubles the output size
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        lstm_out, _ = self.lstm(x)  # lstm_out shape: (batch_size, seq_len, hidden_size * 2)
        
        # Take the last time step's output from both directions
        # and concatenate them before passing to the fully connected layers
        out = self.fc(lstm_out[:, -1, :])
        return out
