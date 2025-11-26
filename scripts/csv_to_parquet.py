import pandas as pd
import os

# Đường dẫn (Path)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Đảm bảo tên file khớp với máy bạn (Telco_customer_churn.xlsx)
INPUT_FILE = os.path.join(BASE_DIR, 'data', 'raw', 'Telco_customer_churn.xlsx')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'parquet', 'raw')


def convert():
    print(f"Reading from {INPUT_FILE}...")

    try:
        # Đọc file Excel
        df = pd.read_excel(INPUT_FILE, engine='openpyxl')
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file tại {INPUT_FILE}")
        return

    # --- ĐOẠN CODE SỬA LỖI Ở ĐÂY ---
    # Tìm cột Total Charges (có thể có khoảng trắng trong tên cột)
    # Chúng ta ép kiểu nó sang số, gặp lỗi (khoảng trắng) thì biến thành NaN

    target_col = 'Total Charges'
    # Kiểm tra xem tên cột trong file là 'Total Charges' hay 'TotalCharges'
    if target_col not in df.columns and 'TotalCharges' in df.columns:
        target_col = 'TotalCharges'

    if target_col in df.columns:
        print(f"Processing column: {target_col}...")
        # errors='coerce' sẽ biến chữ ' ' thành NaN
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
        # Điền 0 vào những chỗ bị NaN
        df[target_col] = df[target_col].fillna(0)
    # -------------------------------

    # Tạo thư mục output nếu chưa có
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Lưu Parquet
    output_path = os.path.join(OUTPUT_DIR, 'telco_churn.parquet')
    try:
        df.to_parquet(output_path, index=False)
        print(f"Success! Saved to {output_path}")
    except Exception as e:
        print(f"Error saving parquet: {e}")


if __name__ == "__main__":
    convert()
