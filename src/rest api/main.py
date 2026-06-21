import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(PROJECT_ROOT)
sys.path.append(current_dir)

import pickle
import sqlite3
import subprocess
import threading
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional
from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Query, BackgroundTasks
from fastapi.responses import FileResponse
from sklearn.preprocessing import MinMaxScaler

from torch.utils.data import DataLoader

import database
import security
import schemas
from src.tryning.time_series_dataset import MedicationTimeSeriesDataset
from src.enging.medication_demand_predictor import MedicationDemandPredictor
from src.enging.restocking_decision_engine import RestockingDecisionEngine

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    database.initialize_database()
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALERS_PATH):
        initialize_model_to_default_state()
    print("\n" + "="*60)
    print("[API] Medication Demand API is running successfully!")
    print("[DOC] Swagger UI Documentation: http://127.0.0.1:8000/docs")
    print("="*60 + "\n")
    yield

app = FastAPI(
    title="Medication Demand Prediction API",
    description="REST API for forecasting medication demand and optimizing inventory replenishment recommendations.",
    version="1.0.0",
    lifespan=lifespan
)

RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "resources", "data", "raw", "Hospital Medication Demand.csv")
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "resources", "data", "processed", "training_data.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "resources", "models", "medication_demand_lstm.pth")
SCALERS_PATH = os.path.join(PROJECT_ROOT, "resources", "models", "scalers.pkl")

def initialize_model_to_default_state():
    model = MedicationDemandPredictor(
        input_size=4,
        hidden_size=64,
        num_layers=2,
        forecast_horizon=7
    )
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    
    if os.path.exists(PROCESSED_DATA_PATH):
        dataframe = pd.read_csv(PROCESSED_DATA_PATH)
    elif os.path.exists(RAW_DATA_PATH):
        dataframe = pd.read_csv(RAW_DATA_PATH)
        dataframe["date"] = pd.to_datetime(dataframe["date"])
        dataframe["month"] = dataframe["date"].dt.month
        dataframe["dayofweek"] = dataframe["date"].dt.dayofweek
        dataframe["is_weekend"] = dataframe["dayofweek"].apply(lambda x: 1 if x >= 5 else 0)
    else:
        dataframe = pd.DataFrame({
            "total_quantity": [0, 100],
            "month": [1, 12],
            "dayofweek": [0, 6],
            "is_weekend": [0, 1]
        })
        
    feature_columns = ["total_quantity", "month", "dayofweek", "is_weekend"]
    scalers = {}
    for column in feature_columns:
        scaler = MinMaxScaler()
        scaler.fit(dataframe[[column]])
        scalers[column] = scaler
        
    with open(SCALERS_PATH, "wb") as file:
        pickle.dump(scalers, file)

training_state = {
    "status": "idle",
    "current_epoch": 0,
    "epochs_total": 0,
    "loss_history": []
}

pipeline_state = {
    "status": "idle",
    "error": None
}



@app.post("/auth/register", response_model=schemas.UserResponse, status_code=status.HTTP_201_CREATED)
def register(user_data: schemas.UserRegister):
    hashed = security.hash_password(user_data.password)
    key = security.generate_api_key()
    
    try:
        with database.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO users (email, full_name, hashed_password, api_key) VALUES (?, ?, ?, ?)",
                (user_data.email, user_data.full_name, hashed, key)
            )
            conn.commit()
            
            cursor.execute("SELECT * FROM users WHERE email = ?", (user_data.email,))
            user = cursor.fetchone()
            
        return {
            "email": user["email"],
            "full_name": user["full_name"],
            "api_key": user["api_key"],
            "created_at": user["created_at"]
        }
    except sqlite3.IntegrityError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

@app.post("/auth/refresh-key", response_model=schemas.ApiKeyResponse)
def refresh_key(credentials: schemas.TokenRefresh):
    with database.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email = ?", (credentials.email,))
        user = cursor.fetchone()
        
    if not user or not security.verify_password(credentials.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
        
    new_key = security.generate_api_key()
    with database.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE users SET api_key = ? WHERE email = ?", (new_key, credentials.email))
        conn.commit()
        
    return {"api_key": new_key}

@app.get("/data/raw", response_class=FileResponse)
def get_raw_data(current_user = Depends(security.get_current_user)):
    if not os.path.exists(RAW_DATA_PATH):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Raw data file not found")
    return FileResponse(RAW_DATA_PATH, media_type="text/csv", filename="raw_medication_demand.csv")

@app.post("/data/raw/upload")
def upload_raw_data(
    file: UploadFile = File(...),
    current_user = Depends(security.get_current_user)
):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file must be a CSV")
        
    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Error parsing CSV: {str(e)}")
        
    required_columns = {"article_id", "date", "total_quantity"}
    if not required_columns.issubset(df.columns):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"CSV must contain exactly these columns: {', '.join(required_columns)}"
        )
        
    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
    
    if os.path.exists(RAW_DATA_PATH):
        df.to_csv(RAW_DATA_PATH, mode="a", header=False, index=False)
    else:
        df.to_csv(RAW_DATA_PATH, index=False)
        
    return {"detail": f"Successfully appended {len(df)} records to raw data"}

@app.get("/data/processed", response_class=FileResponse)
def get_processed_data(current_user = Depends(security.get_current_user)):
    if not os.path.exists(PROCESSED_DATA_PATH):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Processed training data not found")
    return FileResponse(PROCESSED_DATA_PATH, media_type="text/csv", filename="processed_training_data.csv")

def run_pipeline_worker():
    global pipeline_state
    pipeline_state["status"] = "processing"
    pipeline_state["error"] = None
    try:
        subprocess.run([sys.executable, os.path.join(PROJECT_ROOT, "run_pipeline.py")], check=True, cwd=PROJECT_ROOT)
        pipeline_state["status"] = "completed"
    except Exception as e:
        pipeline_state["status"] = "failed"
        pipeline_state["error"] = str(e)

@app.post("/data/process")
def trigger_pipeline(
    background_tasks: BackgroundTasks,
    current_user = Depends(security.get_current_user)
):
    global pipeline_state
    if pipeline_state["status"] == "processing":
        return {"detail": "Pipeline is already running"}
        
    background_tasks.add_task(run_pipeline_worker)
    return {"detail": "Data processing pipeline started in background"}

@app.delete("/data/processed")
def clear_processed_data(current_user = Depends(security.get_current_user)):
    if os.path.exists(PROCESSED_DATA_PATH):
        os.remove(PROCESSED_DATA_PATH)
        return {"detail": "Processed training data cleared successfully"}
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Processed training data not found")

def train_model_worker(epochs):
    global training_state
    training_state["status"] = "training"
    training_state["current_epoch"] = 0
    training_state["epochs_total"] = epochs
    training_state["loss_history"] = []
    
    try:
        dataframe = pd.read_csv(PROCESSED_DATA_PATH)
        feature_columns = ["total_quantity", "month", "dayofweek", "is_weekend"]
        target_column = "total_quantity"
        
        scalers = {}
        for column in feature_columns:
            scaler = MinMaxScaler()
            dataframe[column] = scaler.fit_transform(dataframe[[column]])
            scalers[column] = scaler
            
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        with open(SCALERS_PATH, "wb") as file:
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
        
        if os.path.exists(MODEL_PATH):
            try:
                model.load_state_dict(torch.load(MODEL_PATH))
            except:
                pass
                
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        for epoch in range(epochs):
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
            
            epoch_val_loss = val_loss / len(val_dataset)
            training_state["current_epoch"] = epoch + 1
            training_state["loss_history"].append(float(epoch_val_loss))
            print(f"[TRAINING] Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss / len(train_dataset):.6f} - Val Loss: {epoch_val_loss:.6f}", flush=True)
            
        torch.save(model.state_dict(), MODEL_PATH)
        training_state["status"] = "completed"
    except Exception as e:
        training_state["status"] = f"failed: {str(e)}"

@app.post("/model/train")
def train_model(
    background_tasks: BackgroundTasks,
    epochs: int = Query(10, ge=1),
    current_user = Depends(security.get_current_user)
):
    global training_state
    if training_state["status"] == "training":
        return {"detail": "Model is already training"}
        
    if not os.path.exists(PROCESSED_DATA_PATH):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Processed data missing. Run data pipeline first."
        )
        
    background_tasks.add_task(train_model_worker, epochs)
    return {"detail": f"Model training initiated for {epochs} epochs in background"}

@app.get("/model/status", response_model=schemas.TrainingStatusResponse)
def get_model_status(current_user = Depends(security.get_current_user)):
    global training_state
    return training_state

@app.get("/model/metrics")
def get_model_metrics(current_user = Depends(security.get_current_user)):
    global training_state
    return {
        "status": training_state["status"],
        "loss_history": training_state["loss_history"]
    }

@app.delete("/model")
def clear_model(current_user = Depends(security.get_current_user)):
    initialize_model_to_default_state()
    
    global training_state
    training_state = {
        "status": "idle",
        "current_epoch": 0,
        "epochs_total": 0,
        "loss_history": []
    }
    
    return {"detail": "Model weights, scalers, and status history reset to default initialization states successfully"}

def get_inference_predictions(article_id: str):
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALERS_PATH):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Model not trained. Train the model first."
        )
        
    if not os.path.exists(RAW_DATA_PATH):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Raw data file not found")
        
    df = pd.read_csv(RAW_DATA_PATH)
    df_art = df[df["article_id"] == article_id].copy()
    if len(df_art) < 14:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Insufficient history for this article (minimum 14 days required). Found {len(df_art)} days."
        )
        
    df_art = df_art.sort_values("date").tail(14).copy()
    df_art["date"] = pd.to_datetime(df_art["date"])
    df_art["month"] = df_art["date"].dt.month
    df_art["dayofweek"] = df_art["date"].dt.dayofweek
    df_art["is_weekend"] = df_art["dayofweek"].apply(lambda x: 1 if x >= 5 else 0)
    
    with open(SCALERS_PATH, "rb") as file:
        scalers = pickle.load(file)
        
    feature_columns = ["total_quantity", "month", "dayofweek", "is_weekend"]
    sequence_data = []
    for col in feature_columns:
        val = df_art[[col]].copy()
        val_scaled = scalers[col].transform(val)
        sequence_data.append(val_scaled)
        
    sequence_data = np.hstack(sequence_data)
    input_tensor = torch.tensor(sequence_data, dtype=torch.float32).unsqueeze(0)
    
    model = MedicationDemandPredictor(
        input_size=4,
        hidden_size=64,
        num_layers=2,
        forecast_horizon=7
    )
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    
    with torch.no_grad():
        pred_scaled = model(input_tensor).squeeze(0).numpy()
        
    pred_real = scalers["total_quantity"].inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    pred_real = np.maximum(0.0, pred_real)
    return [int(round(x)) for x in pred_real]

@app.post("/predict/demand", response_model=schemas.PredictDemandResponse)
def predict_demand(
    request: schemas.PredictDemandRequest,
    current_user = Depends(security.get_current_user)
):
    predictions = get_inference_predictions(request.article_id)
    return {
        "article_id": request.article_id,
        "predicted_demand": predictions
    }

@app.post("/predict/restock", response_model=schemas.RestockResponse)
def predict_restock(
    request: schemas.RestockRequest,
    current_user = Depends(security.get_current_user)
):
    predictions = get_inference_predictions(request.article_id)
    
    engine = RestockingDecisionEngine(lead_time=request.lead_time, safety_stock=request.safety_stock)
    recommendation = engine.generate_recommendation(
        current_inventory=request.current_inventory,
        predicted_demand=predictions
    )
    
    return {
        "article_id": request.article_id,
        "demand_during_lead_time": recommendation["demand_during_lead_time"],
        "reorder_point": recommendation["reorder_point"],
        "restock_recommended": recommendation["restock_recommended"],
        "recommended_quantity": recommendation["recommended_quantity"],
        "predicted_demand": predictions
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True, app_dir=current_dir)
