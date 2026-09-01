from fastapi import FastAPI
from pydantic import BaseModel, Field
from src.model_service import HousePriceModel
app = FastAPI(
    title="House Price Prediction API",
    description="API for predictiong California house values",
    version="1.0.0",
)
model =HousePriceModel()
class HouseInput(BaseModel):
    MedInc:float = Field(..., gt=0)
    HouseAge:float = Field(..., gt=0)
    AveRooms:float = Field(..., gt=0)
    Population:float = Field(..., gt=0)
    AveOccup:float = Field(..., gt=0)
    AveBedrms:float = Field(..., gt=0)
    Latitude:float
    Longitude:float
@app.get("/")
def root():
    return {
        "message":"House Price Prediction API is running"
    }
@app.get("/health")
def health():
    return {
        "status":"healthy"
    }
@app.post("/predict")
def predict(house:HouseInput):
    prediction = model.predict(
        house.model_dump()
    )
    return {
        "predicted_house_value": round(prediction, 3),
        "estimated_price_usd": round(prediction * 100000, 2),
    }