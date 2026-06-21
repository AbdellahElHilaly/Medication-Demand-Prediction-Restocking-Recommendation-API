from pydantic import BaseModel, Field
from typing import List, Optional

class UserRegister(BaseModel):
    email: str
    full_name: str = Field(..., min_length=1)
    password: str = Field(..., min_length=6)

class UserResponse(BaseModel):
    email: str
    full_name: str
    api_key: str
    created_at: str

class TokenRefresh(BaseModel):
    email: str
    password: str

class ApiKeyResponse(BaseModel):
    api_key: str

class PredictDemandRequest(BaseModel):
    article_id: str

class PredictDemandResponse(BaseModel):
    article_id: str
    predicted_demand: List[int]

class RestockRequest(BaseModel):
    article_id: str
    current_inventory: int
    lead_time: int = 3
    safety_stock: int = 50

class RestockResponse(BaseModel):
    article_id: str
    demand_during_lead_time: float
    reorder_point: float
    restock_recommended: bool
    recommended_quantity: int
    predicted_demand: List[int]

class TrainingStatusResponse(BaseModel):
    status: str
    current_epoch: int
    epochs_total: int
    loss_history: List[float]
