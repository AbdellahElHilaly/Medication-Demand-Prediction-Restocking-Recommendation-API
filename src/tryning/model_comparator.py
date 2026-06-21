import os
import sys
import pickle
import argparse
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append("C:/src/deep learning")

from src.enging.medication_demand_predictor import MedicationDemandPredictor

def compare_model_and_real(article_id, num_days=100):
    with open("resources/models/scalers.pkl", "rb") as file:
        scalers = pickle.load(file)
        
    model = MedicationDemandPredictor(
        input_size=4,
        hidden_size=64,
        num_layers=2,
        forecast_horizon=7
    )
    model.load_state_dict(torch.load("resources/models/medication_demand_lstm.pth"))
    model.eval()
    
    df = pd.read_csv("resources/data/processed/training_data.csv")
    article_df = df[df["article_id"] == article_id].sort_values("date").copy()
    
    if len(article_df) < 14 + num_days:
        num_days = len(article_df) - 14
        
    feature_columns = ["total_quantity", "month", "dayofweek", "is_weekend"]
    
    real_values = []
    predicted_values = []
    dates = []
    
    for i in range(len(article_df) - num_days, len(article_df)):
        sequence_df = article_df.iloc[i - 14 : i].copy()
        
        sequence_data = []
        for col in feature_columns:
            val = sequence_df[[col]].copy()
            val_scaled = scalers[col].transform(val)
            sequence_data.append(val_scaled)
            
        sequence_data = np.hstack(sequence_data)
        input_tensor = torch.tensor(sequence_data, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            pred_scaled = model(input_tensor).squeeze(0).numpy()
            
        pred_real = scalers["total_quantity"].inverse_transform(pred_scaled[0].reshape(-1, 1))[0, 0]
        pred_real = max(0.0, pred_real)
        
        real_val = article_df.iloc[i]["total_quantity"]
        date_val = article_df.iloc[i]["date"]
        
        real_values.append(real_val)
        predicted_values.append(pred_real)
        dates.append(date_val)
        
    plt.figure(figsize=(12, 6))
    plt.plot(dates, real_values, label="Real Demand", color="blue", linestyle="-", linewidth=2)
    plt.plot(dates, predicted_values, label="Predicted Demand", color="red", linestyle="--", linewidth=2)
    
    plt.title(f"Model Comparison vs Real Demand for {article_id}")
    plt.xlabel("Date")
    plt.ylabel("Quantity")
    plt.xticks(ticks=range(0, len(dates), max(1, len(dates)//10)), labels=dates[::max(1, len(dates)//10)], rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    os.makedirs("resources/data/processed/plots", exist_ok=True)
    plot_path = f"resources/data/processed/plots/model_comparison_{article_id}.png"
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Saved comparison plot to {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--article", type=str, default="ART001")
    parser.add_argument("--days", type=int, default=100)
    args = parser.parse_args()
    
    compare_model_and_real(args.article, args.days)
