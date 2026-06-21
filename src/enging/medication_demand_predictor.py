import torch
import torch.nn as nn

class MedicationDemandPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, forecast_horizon):
        super(MedicationDemandPredictor, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, forecast_horizon)

    def forward(self, x):
        lstm_out, (hn, cn) = self.lstm(x)
        last_hidden_state = lstm_out[:, -1, :]
        out = self.fc(last_hidden_state)
        return out
