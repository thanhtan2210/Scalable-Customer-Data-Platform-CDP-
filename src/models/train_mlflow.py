import pandas as pd
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- CẤU HÌNH ---
INPUT_PATH = "s3://datalake/processed/features"

# Cấu hình MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("CDP_Churn_Prediction")

# Cấu hình MinIO cho MLflow
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "admin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "password"


def train():
    print("--- Starting Training with MLflow Tracking ---")

    # 1. Load Data
    try:
        print(f"🚀 Reading data from: {INPUT_PATH}")
        df = pd.read_parquet(
            INPUT_PATH,
            storage_options={
                "key": "admin",
                "secret": "password",
                "client_kwargs": {"endpoint_url": "http://localhost:9000"}
            }
        )
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # 2. Prepare Data
    if 'Churn' not in df.columns:
        print("❌ Error: Column 'Churn' not found.")
        return

    X = df.drop(columns=['customerID', 'Churn'])
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # 3. START MLFLOW RUN
    with mlflow.start_run():
        print("🧪 Experiment started...")

        # A. Log Params
        n_estimators = 100
        max_depth = 10
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("data_source", INPUT_PATH)

        # B. Train
        print("Training Random Forest...")
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)

        # C. Evaluate
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"✅ Model Accuracy: {acc:.4f}")
        mlflow.log_metric("accuracy", acc)

        # D. Log Model (SỬA ĐOẠN NÀY ĐỂ HẾT WARNING)
        print("💾 Saving model to MLflow/MinIO...")

        # Dùng keyword arguments rõ ràng
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="s3://mlflow/",
            name="random_forest_model",
            registered_model_name="TelcoChurnModel"
        )

        print(f"✨ Done! View results at http://localhost:5000")


if __name__ == "__main__":
    train()
