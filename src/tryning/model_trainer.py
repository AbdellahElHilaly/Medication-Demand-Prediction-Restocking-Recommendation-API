import os
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.tryning.time_series_dataset import MedicationTimeSeriesDataset
from src.enging.medication_demand_predictor import MedicationDemandPredictor

def train_model():
    dataframe = pd.read_csv("resources/data/processed/training_data.csv")
    
    feature_columns = ["total_quantity", "month", "dayofweek", "is_weekend"]
    target_column = "total_quantity"
    
    scalers = {}
    for column in feature_columns:
        scaler = MinMaxScaler()
        dataframe[column] = scaler.fit_transform(dataframe[[column]])
        scalers[column] = scaler
        
    os.makedirs("resources/models", exist_ok=True)
    with open("resources/models/scalers.pkl", "wb") as file:
        pickle.dump(scalers, file)
        
    sequence_length = 14
    forecast_horizon = 7
    
    dataset = MedicationTimeSeriesDataset(
        dataframe=dataframe,
        sequence_length=sequence_length,
        forecast_horizon=forecast_horizon,
        feature_columns=feature_columns,
        target_column=target_column
    )
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    model = MedicationDemandPredictor(
        input_size=len(feature_columns),
        hidden_size=64,
        num_layers=2,
        forecast_horizon=forecast_horizon
    )
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(10):
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(x_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x_batch.size(0)
            
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                predictions = model(x_batch)
                loss = criterion(predictions, y_batch)
                val_loss += loss.item() * x_batch.size(0)
                
        print(f"Epoch {epoch+1}/10 - Train Loss: {train_loss/len(train_dataset):.6f} - Val Loss: {val_loss/len(val_dataset):.6f}")
        
    torch.save(model.state_dict(), "resources/models/medication_demand_lstm.pth")

if __name__ == "__main__":
    train_model()
