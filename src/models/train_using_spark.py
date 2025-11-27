import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Setup đường dẫn
BASE_DIR = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

# Đường dẫn MinIO
# LƯU Ý QUAN TRỌNG: Pandas dùng s3:// chứ không dùng s3a://
INPUT_PATH = "s3://datalake/processed/features"
MODEL_DIR = os.path.join(BASE_DIR, 'models')


def train():
    print("--- Starting Training Job (MinIO Version) ---")

    # 1. Load Data trực tiếp từ MinIO
    try:
        print(f"🚀 Reading data from MinIO: {INPUT_PATH}")

        # Pandas tự động dùng s3fs để đọc S3 thông qua storage_options
        df = pd.read_parquet(
            INPUT_PATH,
            storage_options={
                "key": "admin",
                "secret": "password",
                "client_kwargs": {"endpoint_url": "http://localhost:9000"}
            }
        )
        print(f"✅ Loaded {len(df)} rows.")
    except Exception as e:
        print(f"❌ Lỗi đọc file từ MinIO: {e}")
        print("💡 Gợi ý: Kiểm tra xem Docker MinIO có đang chạy không?")
        print("💡 Gợi ý: Kiểm tra xem Spark Job đã ghi file vào 'datalake/processed/features' chưa?")
        return

    # 2. Prepare X, y
    if 'Churn' not in df.columns:
        print(
            f"ERROR: Không tìm thấy cột 'Churn'. Các cột hiện có: {list(df.columns)}")
        return

    X = df.drop(columns=['customerID', 'Churn'])
    y = df['Churn']

    print(f"Features used for training: {list(X.columns)}")

    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # 4. Train
    print("Training Random Forest...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 5. Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Model Accuracy: {acc:.4f}")

    # 6. Save Model Local (Sau này có thể nâng cấp save lên MLflow)
    os.makedirs(MODEL_DIR, exist_ok=True)
    save_path = os.path.join(MODEL_DIR, 'churn_model.joblib')
    joblib.dump(model, save_path)
    print(f"💾 Model saved locally to: {save_path}")


if __name__ == "__main__":
    train()
