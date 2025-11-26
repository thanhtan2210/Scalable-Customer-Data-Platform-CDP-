# Data Cleaning Guide for Telco Customer Churn Dataset

This document describes a practical, step-by-step process to clean the Telco customer churn dataset used in this project. The guide includes explanations in both English and Vietnamese, ready-to-run code snippets, common pitfalls, and recommendations for reproducible cleaning.

File references
- Dataset used in this notebook: `Telco_customer_churn.xlsx`
- Notebook: `clean_EDA.ipynb`

---

## 1. Overview / Tổng quan

- Purpose: convert raw dataset into a clean, analysis-ready table suitable for exploratory data analysis and modeling.
- Mục tiêu: biến dữ liệu thô thành bảng sạch, sẵn sàng cho EDA và modeling.

- Meaning / Tác dụng: Establishes the main goals and boundaries of the cleaning work so you can judge whether a change is appropriate for analysis or modeling. It helps prioritize tasks (e.g., fix types, remove duplicates) and communicate expected outcomes to stakeholders.
- Ý nghĩa / Tác dụng: Xác định mục tiêu chính và phạm vi của việc làm sạch, giúp bạn quyết định sửa đổi nào cần thiết cho phân tích hoặc mô hình, ưu tiên các bước (chuyển kiểu, loại bỏ trùng lặp...) và truyền đạt kết quả mong muốn.

Tasks covered (high-level)
- Read and inspect the file
- Normalize column names and select relevant columns
- Convert types (numeric, dates)
- Handle missing values and duplicates
- Clean categorical values and encode
- Detect and handle outliers
- Basic feature engineering
- Save cleaned dataset

---

## 2. Prerequisites / Yêu cầu môi trường

- Python 3.8+ and common data packages: `pandas`, `numpy`, `matplotlib`, `seaborn`.
- Optional for modeling/statistics: `scipy`, `scikit-learn`.

Install recommended packages (Windows PowerShell):

```powershell
python -m pip install --upgrade pip setuptools wheel
python -m pip install pandas numpy matplotlib seaborn scipy scikit-learn
```

If you use conda:

```powershell
conda install -c conda-forge pandas numpy matplotlib seaborn scipy scikit-learn
```

Make sure VS Code/Jupyter uses the same interpreter/environment you installed packages into.

- Meaning / Tác dụng: Ensures reproducibility — having the right packages in the same environment prevents "module not found" or version mismatch errors when running cells. This step reduces friction when sharing the notebook.
- Ý nghĩa / Tác dụng: Đảm bảo tính tái tạo — cài đúng gói vào môi trường đang dùng tránh lỗi thiếu module hoặc phiên bản, giảm rào cản khi chia sẻ notebook.

---

## 3. Read the data / Đọc dữ liệu

Python snippet:

```python
import pandas as pd
df = pd.read_excel('Telco_customer_churn.xlsx')
df.shape
df.head()
df.dtypes
```

What to check:
- Is the file located in the same folder as the notebook? If not, provide correct relative or absolute path.
- Check the number of rows/columns and a few top rows.

- Meaning / Tác dụng: A quick initial check helps spot formatting issues, wrong sheets, or encoding problems before spending time on heavier processing. It gives a fast picture of initial data quality.
- Ý nghĩa / Tác dụng: Kiểm tra nhanh ban đầu giúp phát hiện lỗi định dạng, sheet sai hoặc vấn đề encoding trước khi làm các bước nặng hơn; cung cấp cái nhìn tổng quan về chất lượng dữ liệu.

---

## 4. Normalize column names and select columns / Chuẩn hóa tên cột

Reasons: whitespace, hidden characters or inconsistent capitalization cause hard-to-find bugs.

```python
# strip whitespace and unify case
df.columns = df.columns.str.strip()
```

Select only relevant columns (example used in this project):

```python
cols = ['CustomerID','City','Gender','Senior Citizen','Partner','Dependents',
        'Tenure Months','Phone Service','Multiple Lines','Internet Service',
        'Online Security','Online Backup','Device Protection','Tech Support',
        'Streaming TV','Streaming Movies','Contract','Paperless Billing',
        'Payment Method','Monthly Charges','Total Charges','Churn Label',
        'Churn Value','Churn Score','CLTV','Churn Reason']
df = df[cols].copy()
```

If a listed column does not exist, inspect `df.columns` and adapt the list.

- Meaning / Tác dụng: Normalizing column names prevents typos and hidden-character bugs. Selecting a focused subset of columns reduces memory usage and keeps the workflow simpler and reproducible.
- Ý nghĩa / Tác dụng: Chuẩn hoá tên cột tránh lỗi gõ và ký tự ẩn; lọc cột cần thiết giảm dùng bộ nhớ và giúp workflow rõ ràng, dễ tái tạo.

---

## 5. Convert types / Chuyển kiểu dữ liệu

- `Total Charges` often imported as object because of stray characters – convert to numeric.

```python
df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')
df['Tenure Months'] = pd.to_numeric(df['Tenure Months'], errors='coerce')
```

- Re-check types with `df.info()`.

- Meaning / Tác dụng: Ensures numeric computations and aggregations work correctly and prevents accidental string arithmetic. Use `df.info()` to verify conversions and find any remaining problematic columns.
- Ý nghĩa / Tác dụng: Đảm bảo các phép toán số và thống kê thực hiện đúng; `df.info()` xác nhận chuyển kiểu thành công và giúp phát hiện cột còn vấn đề.

---

## 6. Missing values strategy / Xử lý giá trị thiếu

Steps:

1. Quantify missingness:

```python
df.isna().sum()
```

2. Decide per-column action:
- If a column critical for analysis (e.g. `Total Charges`) has very few NaNs → drop those rows.
- If a column numeric with moderate missing → impute (median) or model-based imputation.
- If a categorical column → treat NaN as a separate category or impute with mode.

Examples:

```python
# drop rows where Total Charges is missing (common in Telco dataset)
df.dropna(subset=['Total Charges'], inplace=True)

# impute Tenure with median
df['Tenure Months'] = df['Tenure Months'].fillna(df['Tenure Months'].median())

# categorical fill
df['Internet Service'] = df['Internet Service'].fillna('Unknown')
```

Document any drops/imputations (log counts before/after) for reproducibility.

- Meaning / Tác dụng: The chosen missing-value strategy (drop vs impute) affects bias and model outcomes. Logging counts before/after gives transparency and allows auditing or reverting the decision.
- Ý nghĩa / Tác dụng: Cách xử lý giá trị thiếu ảnh hưởng tới bias và kết quả mô hình; ghi lại số lượng trước/sau giúp minh bạch, dễ kiểm tra và hoàn nguyên khi cần.

---

## 7. Duplicates / Kiểm tra bản sao

```python
dups = df.duplicated().sum()
print('duplicate rows:', dups)
df = df.drop_duplicates()
```

If duplicates exist because of ID duplication, inspect rows and decide which to keep.

- Meaning / Tác dụng: Removing duplicates prevents inflated counts and biased statistics or models. When duplicates reflect different events for the same customer, consider aggregation rather than deletion.
- Ý nghĩa / Tác dụng: Loại bỏ bản sao tránh làm sai lệch thống kê và mô hình; nếu bản sao phản ánh nhiều sự kiện của cùng khách hàng, cân nhắc gộp thay vì xóa.

---

## 8. Clean categorical values / Chuẩn hóa giá trị phân loại

Common tasks:

- Trim whitespace and unify case:

```python
for c in ['Gender','Internet Service','Contract','Payment Method','Churn Label']:
    df[c] = df[c].astype(str).str.strip()
```

- Map binary strings to 0/1 for modeling:

```python
df['Partner'] = df['Partner'].map({'Yes':1,'No':0})
df['Senior Citizen'] = df['Senior Citizen'].map({'Yes':1,'No':0})
df['Dependents'] = df['Dependents'].map({'Yes':1,'No':0})
```

- Use `pd.get_dummies()` for multi-class categoricals when needed for models.

- Meaning / Tác dụng: Cleaning categorical strings (trimming, case) avoids accidental category splits. Encoding (binary/one-hot) converts human-readable labels into numeric features required by most ML algorithms.
- Ý nghĩa / Tác dụng: Chuẩn hóa chuỗi phân loại tránh chia nhãn sai; mã hóa biến phân loại thành dạng số để các thuật toán ML có thể sử dụng.

---

## 9. Outliers / Giá trị ngoại lai

Quick IQR-based capping (winsorization):

```python
def cap_outliers(series):
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5*iqr, q3 + 1.5*iqr
    return series.clip(lower, upper)

df['Monthly Charges'] = cap_outliers(df['Monthly Charges'])
```

Decide whether to cap or remove outliers depending on domain knowledge and modeling needs.

- Meaning / Tác dụng: Outliers can disproportionately influence means and model fits. Capping (winsorizing) reduces impact while preserving observations; removing should be reserved for clear data errors.
- Ý nghĩa / Tác dụng: Giá trị ngoại lai có thể làm sai lệch tham số và mô hình; capping giảm ảnh hưởng nhưng giữ dữ liệu, xóa chỉ dùng khi chắc chắn là lỗi.

---

## 10. Transformations / Biến đổi

- If `Total Charges` has a heavy right-skew, consider log-transform for visualization/modeling.

- Meaning / Tác dụng: Transformations like log1p make skewed distributions more symmetric, stabilize variance, and often improve performance of linear models or distance-based algorithms.
- Ý nghĩa / Tác dụng: Các biến đổi (như log) giảm độ lệch phải, ổn định phương sai và cải thiện hiệu năng của một số mô hình tuyến tính và khoảng cách.

```python
df['TotalCharges_log'] = np.log1p(df['Total Charges'])
```

---

## 11. Feature engineering / Tạo đặc trưng

Examples useful for churn analysis:

- Average monthly spend (guard against division by zero):

```python
df['avg_monthly'] = df['Total Charges'] / df['Tenure Months'].replace({0:1})
```

- Tenure groups:

```python
df['tenure_group'] = pd.cut(df['Tenure Months'], bins=[-1,6,12,24,60,999], labels=['0-6','7-12','13-24','25-60','60+'])
```

---

- Meaning / Tác dụng: New features (e.g., average monthly spend, tenure groups) capture domain-specific signals that raw columns may not express directly, often improving model discriminative power.
- Ý nghĩa / Tác dụng: Tạo đặc trưng mới giúp nắm bắt hành vi khách hàng tốt hơn so với dùng trực tiếp các cột gốc, thường nâng cao khả năng phân biệt của mô hình.

## 12. Encoding for modeling / Mã hóa cho mô hình

- Binary map: `Yes/No` → `1/0`.
- One-hot: `pd.get_dummies(df, columns=[...], drop_first=True)`.
- Label encoding only when categories are ordinal.

Scale numeric features if algorithm requires it (e.g., Logistic Regression, SVM):

```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
num_cols = ['Monthly Charges','Total Charges','Tenure Months','avg_monthly']
df[num_cols] = scaler.fit_transform(df[num_cols])
```

Note: install `scikit-learn` with `python -m pip install scikit-learn` (don't use `pip install sklearn`).

- Meaning / Tác dụng: Proper encoding and scaling ensure numeric features are comparable and that models relying on distances or gradients behave well and converge reliably.
- Ý nghĩa / Tác dụng: Mã hóa và chuẩn hóa giúp các đặc trưng số có thang đo tương đương, làm cho các thuật toán dựa trên khoảng cách/gradient hoạt động ổn định hơn.

---

- Meaning / Tác dụng: Saving the cleaned dataset preserves your preprocessing work so downstream experiments and model training can reuse the cleaned data without re-running expensive steps.
- Ý nghĩa / Tác dụng: Lưu dữ liệu đã làm sạch giữ lại kết quả tiền xử lý, giúp các bước thử nghiệm và huấn luyện mô hình tiếp theo sử dụng lại dữ liệu mà không cần chạy lại toàn bộ pipeline.

## 13. Save cleaned dataset / Lưu dữ liệu đã clean

Preferred formats: Parquet (fast, preserves dtypes) or CSV.

```python
df.to_parquet('Telco_customer_churn_clean.parquet', index=False)
# or
df.to_csv('Telco_customer_churn_clean.csv', index=False)
```

---

## 14. Reproducibility & logging / Đảm bảo tái tạo

- Keep a copy of raw data unmodified.
- Save the notebook with each cleaning step in separate cells with comments.
- Log row counts and missing summaries before/after each major operation.

Example logging pattern:

```python
def log_step(name):
    print(name)
    print('shape:', df.shape)
    print(df.isna().sum())

log_step('after load')
# perform op...
log_step('after dropna Total Charges')

- Meaning / Tác dụng: Logging each major step and its effect on shape/missingness provides an audit trail to explain how data changed, aiding debugging and reproducibility.
- Ý nghĩa / Tác dụng: Ghi lại các bước chính và ảnh hưởng của chúng lên kích thước/dữ liệu thiếu giúp truy xuất lịch sử thay đổi, hỗ trợ debug và tái tạo kết quả.
```

---

## 15. Quick troubleshooting / Vấn đề thường gặp

- "File not found": check working directory and relative path. Use `!pwd` (or `os.getcwd()`).
- "pd.to_numeric" produced many NaNs: check for non-numeric characters (commas, currency symbols).
- Installing `sklearn` failed — use `scikit-learn` package name instead.
- Pairplot or heavy plots slow: sample with `df.sample(1000, random_state=1)`.

- Meaning / Tác dụng: Quick troubleshooting tips save time by addressing common runtime issues (missing files, package errors, heavy plotting) so you can focus on analysis.
- Ý nghĩa / Tác dụng: Mẹo xử lý nhanh giúp giảm thời gian debug các lỗi phổ biến (thiếu file, lỗi cài đặt, vẽ heavy plot) để tập trung vào phân tích.

---

## 16. Suggested notebook cells order (recommended)

1. Imports and read data
2. Quick inspection (shape, head, dtypes)
3. Column normalization and selection
4. Type conversions
5. Missing value handling
6. Duplicates removal
7. Categorical cleaning
8. Numeric EDA (hist/KDE/boxplot)
9. Outlier handling
10. Feature engineering
11. Encoding & scaling
12. Save cleaned data

- Meaning / Tác dụng: The suggested order gives a logical, incremental flow that is easy to follow, test, and revert — improving readability and reproducibility of the notebook.
- Ý nghĩa / Tác dụng: Thứ tự đề xuất tạo luồng công việc tuần tự, dễ kiểm tra và hoàn nguyên, giúp notebook dễ đọc và dễ tái tạo.


