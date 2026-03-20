import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy import stats
import pickle
import warnings
warnings.filterwarnings('ignore')

input_file = 'dữ_liệu_tổng.csv'
output_file = 'dữ_liệu_sạch.csv'
scaler_file = 'scaler.pkl'
encoder_file = 'encoders.pkl'

# Bước 1: Đọc dữ liệu
print("\n[1/14] Đọc dữ liệu...")
df = pd.read_csv(input_file, encoding='utf-8-sig')
print(f"  → Kích thước: {df.shape}")
print(f"  → Cột: {list(df.columns)}")

# Bước 2: Xử lý dữ liệu thiếu
print("\n[2/14] Xử lý dữ liệu thiếu...")
missing = df.isnull().sum()
if missing.any():
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in [np.float64, np.int64]:
                df[col].fillna(df[col].median(), inplace=True)
                print(f"  → {col}: Điền trung vị")
            else:
                mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
                df[col].fillna(mode_val, inplace=True)
                print(f"  → {col}: Điền mode")
else:
    print("  → Không có dữ liệu thiếu")

# Bước 3: Loại bỏ trùng lặp
print("\n[3/14] Loại bỏ dữ liệu trùng lặp...")
before = len(df)
df = df.drop_duplicates()
print(f"  → Xóa: {before - len(df)} dòng trùng lặp")

# Bước 4: Phát hiện ngoại lệ (IQR Method) - CHỈ cho cột số
print("\n[4/14] Phát hiện ngoại lệ (IQR Method)...")
numeric_cols_all = df.select_dtypes(include=[np.number]).columns.tolist()
outlier_info = {}
for col in numeric_cols_all:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    if IQR > 0:  # Tránh chia cho 0
        outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
        if len(outliers) > 0:
            outlier_info[col] = len(outliers)
            print(f"  → {col}: {len(outliers)} ngoại lệ ({len(outliers)/len(df)*100:.2f}%)")

print("  → Quyết định: Giữ lại ngoại lệ (có thể là dữ liệu thực)")

# Bước 5: Mã hóa categorical (thứ, mùa)
print("\n[5/14] Mã hóa các cột phân loại...")
categorical_cols = ['thứ', 'mùa']
encoders_dict = {}
for col in categorical_cols:
    if col in df.columns:
        encoder = LabelEncoder()
        df[col + '_mã'] = encoder.fit_transform(df[col].astype(str))
        encoders_dict[col] = encoder
        print(f"  → {col}: {len(encoder.classes_)} lớp")
    else:
        print(f"  → {col}: Không tồn tại trong dữ liệu")

if encoders_dict:
    pickle.dump(encoders_dict, open(encoder_file, 'wb'))

# Bước 6: Chọn cột quan trọng nhất
print("\n[6/14] Chọn cột quan trọng nhất...")
cols_important = [
    'nhiệt_độ_ngoài', 'độ_ẩm_ngoài', 'áp_suất', 'tốc_độ_gió',
    'bức_xạ_mặt_trời', 'lượng_mưa',
    'nhiệt_độ_trong', 'độ_ẩm_trong',
    'giờ', 'tháng', 'mùa_mã', 'cuối_tuần',
    'chỉ_số_thoải_mái'
]

# Kiểm tra cột nào có trong dữ liệu
cols_available = []
for col in cols_important:
    if col in df.columns:
        cols_available.append(col)
    else:
        print(f"  ⚠️ {col}: Không có trong dữ liệu")

df_selected = df[cols_available].copy()
print(f"  → Giữ {len(cols_available)} cột: {cols_available}")

# Bước 7: Thống kê chi tiết TRƯỚC chuẩn hóa
print("\n[7/14] Thống kê chi tiết TRƯỚC chuẩn hóa...")
numeric_features = df_selected.select_dtypes(include=[np.number]).columns.tolist()
print("\n  Min/Max/Mean/Std:")
for col in numeric_features:
    print(f"    {col:20} | Min: {df_selected[col].min():8.2f} | Max: {df_selected[col].max():8.2f} | Mean: {df_selected[col].mean():8.2f} | Std: {df_selected[col].std():8.2f}")

# Bước 8: Kiểm tra Skewness (độ lệch)
print("\n[8/14] Kiểm tra Skewness (độ lệch dữ liệu)...")
print("  Skewness > 1 hoặc < -1: Dữ liệu lệch, có thể cần Log transformation")
for col in numeric_features:
    if col != 'mùa_mã' and col != 'cuối_tuần':  # Bỏ qua các biến mã hóa
        skewness = stats.skew(df_selected[col].dropna())
        print(f"    {col:20} | Skewness: {skewness:7.3f}", end="")
        if abs(skewness) > 1:
            print(" → Lệch nhiều")
        else:
            print(" → OK")

# Bước 9: Log transformation (nếu skewed)
print("\n[9/14] Log transformation (tùy chọn cho cột lệch)...")
# Chỉ áp dụng cho cột có giá trị dương
skewed_cols = []
for col in numeric_features:
    if col not in ['giờ', 'tháng', 'mùa_mã', 'cuối_tuần']:  # Bỏ qua biến thời gian
        if df_selected[col].min() > 0:  # Log chỉ cho giá trị dương
            skewness = abs(stats.skew(df_selected[col].dropna()))
            if skewness > 1:
                df_selected[col + '_log'] = np.log(df_selected[col])
                skewed_cols.append(col)
                print(f"  → {col}: áp dụng log transformation")

if not skewed_cols:
    print("  → Không có cột nào cần log transformation")

# Bước 10: Biểu đồ phân phối TRƯỚC chuẩn hóa
print("\n[10/14] Biểu đồ phân phối TRƯỚC chuẩn hóa...")
important_numeric = ['nhiệt_độ_ngoài', 'độ_ẩm_ngoài', 'nhiệt_độ_trong', 'độ_ẩm_trong', 'bức_xạ_mặt_trời', 'lượng_mưa']
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for idx, col in enumerate(important_numeric):
    if col in df_selected.columns:
        axes[idx].hist(df_selected[col], bins=30, edgecolor='black', alpha=0.7, color='skyblue')
        axes[idx].set_title(f'Phân phối TRƯỚC: {col}')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Tần suất')
        axes[idx].grid(alpha=0.3)
    else:
        axes[idx].axis('off')
        axes[idx].set_title(f'{col}: Không có dữ liệu')
plt.tight_layout()
plt.show()

# Bước 11: Tách feature (X) và target (y)
print("\n[11/14] Tách feature và target...")
target_col = 'chỉ_số_thoải_mái'
y = df_selected[[target_col]]
X = df_selected.drop([target_col], axis=1)

# Loại bỏ cột _log nếu có
cols_to_drop = [col for col in X.columns if '_log' in col]
if cols_to_drop:
    X = X.drop(cols_to_drop, axis=1)
    print(f"  → Loại bỏ cột log: {cols_to_drop}")

print(f"  → X: {X.shape} | y: {y.shape}")

# Bước 12: Kiểm tra Imbalance (target)
print("\n[12/14] Kiểm tra Imbalance (target distribution)...")
print(f"  Mean: {y[target_col].mean():.2f} | Std: {y[target_col].std():.2f}")
print(f"  Min: {y[target_col].min():.2f} | Max: {y[target_col].max():.2f}")
print("  Distribution: Tương đối cân bằng (không phải classification)")

# Biểu đồ target
plt.figure(figsize=(10, 5))
plt.hist(y[target_col], bins=30, edgecolor='black', alpha=0.7, color='lightcoral')
plt.title(f'Phân phối Target: {target_col}')
plt.xlabel(target_col)
plt.ylabel('Tần suất')
plt.grid(alpha=0.3)
plt.show()

# Bước 13: Feature selection dựa tương quan
print("\n[13/14] Feature selection dựa trên tương quan...")
numeric_X = X.select_dtypes(include=[np.number]).columns.tolist()
correlations = X[numeric_X].corrwith(y[target_col]).sort_values(ascending=False)
print("  Tương quan với target (Top 10):")
for col, corr in correlations.head(10).items():
    print(f"    {col:20} | {corr:7.4f}")

# Loại bỏ cột có tương quan quá thấp (<0.1)
cols_to_drop_low_corr = [col for col, corr in correlations.items() if abs(corr) < 0.1]
if cols_to_drop_low_corr:
    print(f"\n  Cột tương quan thấp (<0.1):")
    for col in cols_to_drop_low_corr:
        print(f"    {col}: {correlations[col]:.4f}")
    print(f"  → Quyết định: Giữ lại (có thể quan trọng cho model)")
else:
    print("  → Tất cả cột đều có tương quan > 0.1")

# Kiểm tra multicollinearity
print("\n  Kiểm tra multicollinearity (tương quan giữa feature):")
corr_matrix = X[numeric_X].corr()
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.8:
            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))

if high_corr_pairs:
    print("  Cặp cột tương quan cao (>0.8):")
    for col1, col2, corr_val in high_corr_pairs:
        print(f"    {col1} - {col2}: {corr_val:.4f}")
else:
    print("  → Không có multicollinearity nghiêm trọng")

# Biểu đồ tương quan
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title("Biểu đồ tương quan giữa các feature")
plt.tight_layout()
plt.show()

# Biểu đồ tương quan với target
fig, ax = plt.subplots(figsize=(10, 6))
correlations.plot(kind='barh', ax=ax, color=['green' if x > 0 else 'red' for x in correlations])
ax.set_xlabel(f'Tương quan với {target_col}')
ax.set_title('Tương quan từng feature với target')
plt.tight_layout()
plt.show()

# Bước 14: Chuẩn hóa (Normalization)
print("\n[14/14] Chuẩn hóa...")
# Chuẩn hóa
scaler = StandardScaler()
X[numeric_X] = scaler.fit_transform(X[numeric_X])
pickle.dump(scaler, open(scaler_file, 'wb'))
print(f"  → Chuẩn hóa {len(numeric_X)} cột số")

# Thống kê SAU chuẩn hóa
print("\n  Min/Max/Mean/Std SAU chuẩn hóa (Mean=0, Std=1):")
for col in numeric_X[:10]:  # Chỉ hiển thị 10 cột đầu
    print(f"    {col:20} | Min: {X[col].min():8.2f} | Max: {X[col].max():8.2f} | Mean: {X[col].mean():8.2f} | Std: {X[col].std():8.2f}")
if len(numeric_X) > 10:
    print(f"    ... và {len(numeric_X)-10} cột khác")

# Biểu đồ phân phối SAU chuẩn hóa
important_numeric = ['nhiệt_độ_ngoài', 'độ_ẩm_ngoài', 'nhiệt_độ_trong', 'độ_ẩm_trong', 'bức_xạ_mặt_trời', 'lượng_mưa']
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for idx, col in enumerate(important_numeric):
    if col in X.columns:
        axes[idx].hist(X[col], bins=30, edgecolor='black', alpha=0.7, color='lightgreen')
        axes[idx].set_title(f'Phân phối SAU: {col} (Normalized)')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Tần suất')
        axes[idx].grid(alpha=0.3)
    else:
        axes[idx].axis('off')
        axes[idx].set_title(f'{col}: Không có dữ liệu')
plt.tight_layout()
plt.show()

# Lưu dữ liệu sạch
df_final = pd.concat([X, y], axis=1)
df_final.to_csv(output_file, index=False, encoding='utf-8-sig')

print("\nTIỀN XỬ LÝ THÀNH CÔNG")
print(f"  Output: {output_file}")
print(f"  Shape: {df_final.shape} | Dòng: {len(df_final):,} | Cột: {df_final.shape[1]}")
print(f"  Files: {scaler_file}, {encoder_file}")
