import numpy as np
import torch
from torch.utils.data import Dataset

class MedicationTimeSeriesDataset(Dataset):
    def __init__(self, dataframe, sequence_length, forecast_horizon, feature_columns, target_column):
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.sequences = []
        self.labels = []
        
        grouped = dataframe.groupby("article_id")
        for article_id, group in grouped:
            sorted_group = group.sort_values("date")
            features = sorted_group[feature_columns].values
            targets = sorted_group[target_column].values
            
            num_samples = len(sorted_group) - sequence_length - forecast_horizon + 1
            for i in range(num_samples):
                x = features[i : i + sequence_length]
                y = targets[i + sequence_length : i + sequence_length + forecast_horizon]
                self.sequences.append(x)
                self.labels.append(y)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return (
            torch.tensor(self.sequences[index], dtype=torch.float32),
            torch.tensor(self.labels[index], dtype=torch.float32)
        )
