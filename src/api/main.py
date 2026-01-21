import os
import pandas as pd
import mlflow.sklearn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

# --- CẤU HÌNH ---
# 1. Cấu hình kết nối MLflow & MinIO
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "admin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "password"
MLFLOW_TRACKING_URI = "http://localhost:5000"

# 2. Tên Model đã đăng ký trong train_mlflow.py
MODEL_NAME = "TelcoChurnModel"
MODEL_STAGE = "None"  # Hoặc "Production" nếu bạn đã set trên UI

# Biến toàn cục để lưu model
ml_models = {}

# --- DATA MODELS ---


class CustomerRequest(BaseModel):
    # Định nghĩa các feature cần thiết để dự đoán
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    # Thêm các feature khác nếu cần


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- LOAD MODEL KHI KHỞI ĐỘNG ---
    print("🔌 Connecting to MLflow...")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    try:
        # Load model phiên bản mới nhất từ MLflow
        model_uri = f"models:/{MODEL_NAME}/1"  # Lấy version 1 (hoặc Latest)
        print(f"📥 Loading model from: {model_uri}")

        model = mlflow.sklearn.load_model(model_uri)
        ml_models["churn_model"] = model
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("⚠️ API sẽ chạy nhưng không thể dự đoán được.")

    yield

    # Clean up
    ml_models.clear()

app = FastAPI(lifespan=lifespan, title="CDP Churn Prediction API")


@app.get("/")
def home():
    return {"message": "CDP API is running with MLflow Integration 🚀"}


@app.post("/predict")
def predict_churn(customer: CustomerRequest):
    if "churn_model" not in ml_models:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Chuyển input thành DataFrame
        input_data = pd.DataFrame([customer.dict()])

        # Dự đoán
        model = ml_models["churn_model"]
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        return {
            "prediction": int(prediction),
            "churn_probability": float(probability),
            "risk_level": "High" if probability > 0.7 else ("Medium" if probability > 0.4 else "Low")
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Prediction error: {str(e)}")
