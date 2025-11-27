import os
import s3fs


def upload_to_minio():
    # 1. Cấu hình kết nối MinIO
    fs = s3fs.S3FileSystem(
        client_kwargs={'endpoint_url': 'http://localhost:9000'},
        key='admin',
        secret='password',
        use_listings_cache=False
    )

    # 2. Đường dẫn
    # Lấy đường dẫn gốc của dự án
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # File local (đã tạo từ bước csv_to_parquet)
    local_path = os.path.join(
        base_dir, 'data', 'parquet', 'raw', 'telco_churn.parquet')

    # Đích đến trên MinIO
    s3_path = 's3://datalake/raw/telco_churn.parquet'

    print(f"⏳ Đang upload từ: {local_path}")
    print(f"➡️ Đến: {s3_path}")

    try:
        if not os.path.exists(local_path):
            print(
                "❌ LỖI: Không tìm thấy file local! Bạn đã chạy 'csv_to_parquet.py' chưa?")
            return

        # Upload
        fs.put(local_path, s3_path)
        print("✅ Upload thành công!")

        # Kiểm tra lại xem file có tồn tại không
        if fs.exists(s3_path):
            print(f"🔍 Đã xác nhận file tồn tại trên MinIO: {s3_path}")
            print(f"📦 Kích thước: {fs.info(s3_path)['size']} bytes")

    except Exception as e:
        print(f"❌ Lỗi: {e}")


if __name__ == "__main__":
    upload_to_minio()
