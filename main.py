from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.model import load_model
from src.preprocess import load_and_clean_data
from src.simulator import predict_sales


app = FastAPI()

# Allow frontend to connect with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Input format for JSON body
class PredictionInput(BaseModel):
    price: float
    discount: float
    promotion: int


# Load model
model = load_model()

# Prepare dataset for column structure
data = load_and_clean_data("data/sales.csv")

X = data.drop(['units_sold', 'date', 'demand_forecast'], axis=1)
X = X.select_dtypes(include=['int64', 'float64'])


@app.get("/")
def home():
    return {"message": "Smart Sales Predictor API running"}


@app.post("/predict")
def predict(input_data: PredictionInput):
    sample = X.iloc[0].copy()

    sample['price'] = input_data.price
    sample['discount'] = input_data.discount
    sample['holiday_promotion'] = input_data.promotion

    prediction = predict_sales(model, sample, X.columns)

    return {"predicted_sales": int(prediction)}