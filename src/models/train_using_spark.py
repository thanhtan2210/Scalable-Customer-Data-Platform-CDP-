import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Setup đường dẫn
BASE_DIR = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))

# SỬA Ở ĐÂY: Trỏ vào THƯ MỤC 'features' thay vì file cụ thể
INPUT_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'features')
MODEL_DIR = os.path.join(BASE_DIR, 'models')


def train():
    print("--- Starting Training Job ---")

    # Kiểm tra đường dẫn tồn tại chưa
    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: Không tìm thấy thư mục data tại {INPUT_PATH}")
        print("Hãy chạy 'python spark_jobs/clean_data_spark.py' trước.")
        return

    # 1. Load Data
    # Pandas read_parquet có thể đọc cả folder chứa nhiều file parquet
    try:
        print(f"Reading data from folder: {INPUT_PATH}")
        df = pd.read_parquet(INPUT_PATH)
        print(f"✅ Loaded {len(df)} rows.")
    except Exception as e:
        print(f"❌ Lỗi đọc file Parquet: {e}")
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

    # 6. Save Model
    os.makedirs(MODEL_DIR, exist_ok=True)
    save_path = os.path.join(MODEL_DIR, 'churn_model.joblib')
    joblib.dump(model, save_path)
    print(f"💾 Model saved to: {save_path}")


if __name__ == "__main__":
    train()
