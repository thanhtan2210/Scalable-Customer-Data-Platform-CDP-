from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import os

app = FastAPI()

# Load model và data vào RAM khi khởi động app
BASE_DIR = os.getcwd()
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'churn_model.joblib')
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'features')

print("Loading resources...")
try:
    model = joblib.load(MODEL_PATH)
    # Load feature store để tra cứu thông tin khách hàng
    features_df = pd.read_parquet(DATA_PATH).set_index("customerID")
    print("System ready!")
except Exception as e:
    print(f"Error loading resources: {e}")


@app.get("/")
def home():
    return {"message": "CDP Churn Prediction API is running"}


@app.post("/predict/{customer_id}")
def predict_churn(customer_id: str):
    # 1. Tìm thông tin khách hàng trong Feature Store
    if customer_id not in features_df.index:
        raise HTTPException(status_code=404, detail="Customer not found")

    # Lấy row dữ liệu của khách đó (bỏ cột Churn nếu có)
    customer_data = features_df.loc[[customer_id]].drop(
        columns=['Churn'], errors='ignore')

    # 2. Dự đoán
    prediction = model.predict(customer_data)[0]
    probability = model.predict_proba(customer_data)[0][1]

    return {
        "customer_id": customer_id,
        "churn_prediction": "Yes" if prediction == 1 else "No",
        "churn_probability": float(probability)
    }
