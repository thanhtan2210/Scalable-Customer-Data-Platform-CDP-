import pandas as pd
import os

# Setup đường dẫn
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(BASE_DIR, 'data', 'parquet',
                          'raw', 'telco_churn.parquet')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'features')


def run_job():
    print("--- Starting ETL Job (Fix for IBM Dataset) ---")

    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: Không tìm thấy file input tại: {INPUT_PATH}")
        return

    # 1. Đọc dữ liệu
    df = pd.read_parquet(INPUT_PATH)
    print(f"Columns ban đầu: {list(df.columns)}")

    # 2. CHUẨN HÓA TÊN CỘT
    # Xóa khoảng trắng và đổi về chữ thường: 'Churn Value' -> 'churnvalue'
    df.columns = [c.strip().lower().replace(' ', '') for c in df.columns]

    # 3. XỬ LÝ TARGET (CHURN) - ĐOẠN QUAN TRỌNG ĐÃ SỬA
    if 'churnvalue' in df.columns:
        # Nếu có cột Churn Value (1/0), dùng luôn
        print("Tìm thấy cột 'churnvalue', đang đổi tên thành 'Churn'...")
        df['Churn'] = df['churnvalue'].fillna(0).astype(int)
    elif 'churnlabel' in df.columns:
        # Nếu chỉ có Churn Label (Yes/No)
        print("Tìm thấy cột 'churnlabel', đang map Yes/No và đổi tên thành 'Churn'...")
        df['Churn'] = df['churnlabel'].map(
            {'Yes': 1, 'No': 0, 'yes': 1, 'no': 0}).fillna(0).astype(int)
    elif 'churn' in df.columns:
        # Trường hợp tên là Churn sẵn
        df['Churn'] = df['churn'].map(
            {'Yes': 1, 'No': 0, 1: 1, 0: 0}).fillna(0).astype(int)
    else:
        print("CẢNH BÁO: Không tìm thấy cột thông tin Churn (Label hoặc Value)!")

    # 4. XỬ LÝ CÁC FEATURES KHÁC
    # Map tên các cột từ file Excel sang tên chuẩn mà Model cần
    # (Lưu ý: Dataset của bạn có thể dùng tên cột hơi khác, code này cố gắng bắt các trường hợp)

    # Xử lý Total Charges
    if 'totalcharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(
            df['totalcharges'], errors='coerce').fillna(0)

    # Xử lý Monthly Charges
    if 'monthlycharges' in df.columns:
        df['MonthlyCharges'] = pd.to_numeric(
            df['monthlycharges'], errors='coerce').fillna(0)

    # Xử lý Tenure (có thể file bạn tên là 'tenuremonths')
    if 'tenuremonths' in df.columns:
        df['tenure'] = df['tenuremonths'].astype(int)
    elif 'tenure' in df.columns:
        df['tenure'] = df['tenure'].astype(int)

    # Đổi tên customerID cho đúng chuẩn
    if 'customerid' in df.columns:
        df['customerID'] = df['customerid']

    # 5. Feature Selection (Chọn lọc)
    required_cols = ['customerID', 'tenure',
                     'MonthlyCharges', 'TotalCharges', 'Churn']

    # Chỉ giữ lại các cột thực sự tồn tại
    final_cols = [c for c in required_cols if c in df.columns]

    print(f"Các cột cuối cùng sẽ lưu: {final_cols}")

    if 'Churn' not in final_cols:
        print("LỖI LỚN: Vẫn chưa tạo được cột Churn. Kiểm tra lại tên cột trong file gốc!")
        return

    df_clean = df[final_cols]

    # 6. Lưu file
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, 'customer_features.parquet')
    df_clean.to_parquet(output_file, index=False)
    print(f"SUCCESS! Saved {len(df_clean)} rows to: {output_file}")


if __name__ == "__main__":
    run_job()
