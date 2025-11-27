import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lower
from pyspark.sql.types import IntegerType, DoubleType

# --- 1. CONFIGURATION & UTILS ---


def setup_windows_env(base_dir):
    """Cấu hình biến môi trường cho Spark trên Windows"""
    # Hadoop
    hadoop_path = os.path.join(base_dir, 'bin', 'hadoop')
    os.environ['HADOOP_HOME'] = hadoop_path

    if not os.path.exists(os.path.join(hadoop_path, 'bin', 'winutils.exe')):
        sys.exit(f"❌ LỖI: Không tìm thấy winutils.exe tại {hadoop_path}")

    # Java (Auto-detect)
    adoptium_dir = os.path.join(base_dir, 'bin', 'Eclipse Adoptium')
    try:
        jdk_name = [f for f in os.listdir(
            adoptium_dir) if f.startswith('jdk-11')][0]
        java_path = os.path.join(adoptium_dir, jdk_name)
        os.environ['JAVA_HOME'] = java_path
    except IndexError:
        sys.exit("❌ LỖI: Không tìm thấy JDK 11 trong bin/Eclipse Adoptium")

    # Update Path
    os.environ['PATH'] = os.pathsep.join([
        os.path.join(hadoop_path, 'bin'),
        os.path.join(java_path, 'bin'),
        os.environ['PATH']
    ])
    print(f"✅ Env Setup: JAVA_HOME={java_path} | HADOOP_HOME={hadoop_path}")


def get_paths():
    """Trả về đường dẫn Input/Output chuẩn format URI"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    def to_uri(p): return f"file:///{p.replace(os.sep, '/')}"

    return (
        base_dir,
        to_uri(os.path.join(base_dir, 'data', 'parquet',
               'raw', 'telco_churn.parquet')),
        to_uri(os.path.join(base_dir, 'data', 'processed', 'features'))
    )

# --- 2. TRANSFORMATION LOGIC ---


def clean_dataframe(df):
    """Hàm chứa toàn bộ logic làm sạch dữ liệu"""

    # A. Chuẩn hóa tên cột (lower + strip)
    df = df.select([col(c).alias(c.strip().lower().replace(' ', ''))
                   for c in df.columns])
    cols = df.columns

    # B. Xử lý cột Churn (Ưu tiên: churnvalue > churnlabel > churn)
    if 'churnvalue' in cols:
        df = df.withColumn("Churn", col("churnvalue").cast(IntegerType()))
    elif 'churnlabel' in cols:
        df = df.withColumn("Churn", when(
            lower(col("churnlabel")) == "yes", 1).otherwise(0))
    elif 'churn' in cols:
        df = df.withColumn("Churn", when(
            col("churn").isin("Yes", "1"), 1).otherwise(0))

    df = df.fillna(0, subset=["Churn"])

    # C. Ép kiểu số (Numeric Casting)
    numeric_cols = {
        'totalcharges': 'TotalCharges',
        'monthlycharges': 'MonthlyCharges'
    }
    for src, dest in numeric_cols.items():
        if src in cols:
            df = df.withColumn(dest, col(src).cast(
                DoubleType())).fillna(0.0, subset=[dest])

    # D. Xử lý Tenure
    tenure_col = 'tenuremonths' if 'tenuremonths' in cols else (
        'tenure' if 'tenure' in cols else None)
    if tenure_col:
        df = df.withColumn("tenure", col(tenure_col).cast(IntegerType()))

    # E. Đổi tên ID
    if 'customerid' in cols:
        df = df.withColumnRenamed("customerid", "customerID")

    # F. Chọn lọc cột cuối cùng
    required = ['customerID', 'tenure',
                'MonthlyCharges', 'TotalCharges', 'Churn']
    final_cols = [c for c in required if c in df.columns]

    return df.select(*final_cols)

# --- 3. MAIN EXECUTION ---


def run():
    base_dir, input_path, output_path = get_paths()

    # 1. Setup Environment
    setup_windows_env(base_dir)

    # 2. Start Spark
    spark = SparkSession.builder \
        .appName("CDP_Telco_ETL") \
        .master("local[*]") \
        .config("spark.sql.caseSensitive", "false") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    # 3. Run ETL
    print(f"🚀 Reading: {input_path}")
    try:
        df = spark.read.parquet(input_path)
        df_clean = clean_dataframe(df)

        print(f"💾 Writing to: {output_path}")
        df_clean.coalesce(1).write.mode("overwrite").parquet(output_path)
        print("✅ SUCCESS! Spark Job Finished.")

    except Exception as e:
        print(f"❌ ERROR: {e}")
    finally:
        spark.stop()


if __name__ == "__main__":
    run()
