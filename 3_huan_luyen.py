import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import warnings
import os

warnings.filterwarnings('ignore')

# Cấu hình
INPUT_FILE = 'dữ_liệu_sạch.csv'
SCALER_FILE = 'scaler.pkl'
TARGET_COL = 'chỉ_số_thoải_mái'
MODEL_DIR = 'models'

print("\nHUẤN LUYỆN MÔ HÌNH DỰ BÁO CHỈ SỐ THOẢI MÁI")

# Bước 1: Đọc dữ liệu và chuẩn bị
print("\n[1/6] Đọc dữ liệu và chuẩn bị...")
try:
    df = pd.read_csv(INPUT_FILE, encoding='utf-8-sig')
    scaler = pickle.load(open(SCALER_FILE, 'rb'))
    print(f"  → Dữ liệu: {df.shape}")
    print(f"  → Cột: {list(df.columns)}")
except FileNotFoundError as e:
    print(f"   Lỗi: {e}")
    exit()

# Tách feature và target
features = [col for col in df.columns if col != TARGET_COL]
X = df[features].copy()
y = df[TARGET_COL].copy()

print(f"  → Feature: {X.shape[1]} cột: {list(X.columns)}")
print(f"  → Target: {TARGET_COL}")
print(f"  → Sample: {len(X):,} điểm")
print(f"  → Target Range: {y.min():.1f}°C - {y.max():.1f}°C")

# Bước 2: Chia train/test (80/20)
print("\n[2/6] Chia dữ liệu train/test (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)
print(f"  → Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")
print(f"  → Train target - Mean: {y_train.mean():.2f}, Std: {y_train.std():.2f}")
print(f"  → Test target - Mean: {y_test.mean():.2f}, Std: {y_test.std():.2f}")

# Bước 3: Huấn luyện các mô hình
print("\n[3/6] Huấn luyện các mô hình...")
models_dict = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2
    )
}

results = {}
for name, model in models_dict.items():
    print(f"  → Huấn luyện {name}...")
    model.fit(X_train, y_train)
    
    # Dự báo trên test set
    y_pred = model.predict(X_test)
    
    # Tính metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {
        'model': model,
        'y_pred': y_pred,
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'y_test': y_test.values
    }
    
    print(f"      MAE: {mae:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}")

# Bước 4: So sánh và chọn mô hình tốt nhất
print("\n[4/6] So sánh mô hình...")
comparison_df = pd.DataFrame({
    'Mô hình': list(results.keys()),
    'MAE': [results[m]['MAE'] for m in results.keys()],
    'RMSE': [results[m]['RMSE'] for m in results.keys()],
    'R²': [results[m]['R2'] for m in results.keys()]
})
print("\n" + comparison_df.to_string(index=False))

# Chọn mô hình tốt nhất (RMSE thấp nhất)
best_model_name = min(results, key=lambda x: results[x]['RMSE'])
best_model = results[best_model_name]['model']
best_results = results[best_model_name]
y_pred_best = best_results['y_pred']

print(f"\n   Mô hình tốt nhất: {best_model_name}")
print(f"     RMSE: {best_results['RMSE']:.4f} | MAE: {best_results['MAE']:.4f} | R²: {best_results['R2']:.4f}")

# Đánh giá thêm
print(f"\n   Đánh giá hiệu suất:")
print(f"     - Sai số trung bình (MAE): {best_results['MAE']:.2f}°C")
print(f"     - Độ chính xác (R²): {best_results['R2']:.2%}")
print(f"     - Sai số tối đa dự kiến: ±{best_results['RMSE']*2:.2f}°C (95% confidence)")

# Bước 5: Biểu đồ đánh giá
print("\n[5/6] Vẽ biểu đồ đánh giá...")

# Biểu đồ 1: So sánh các metrics
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
metrics = ['MAE', 'RMSE', 'R2']

for idx, metric in enumerate(metrics):
    values = [results[m][metric] for m in results.keys()]
    colors = ['green' if results[name][metric] == min(values) else 'skyblue' for name in results.keys()]
    bars = axes[idx].bar(results.keys(), values, color=colors)
    axes[idx].set_title(f'So sánh {metric}')
    axes[idx].set_ylabel(metric)
    axes[idx].grid(alpha=0.3, axis='y')
    
    # Thêm giá trị lên cột
    for bar, val in zip(bars, values):
        height = bar.get_height()
        axes[idx].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                      f'{val:.4f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Biểu đồ 2: Phân tích phần dư và so sánh dự báo
residuals = best_results['y_test'] - y_pred_best
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scatter: Dự báo vs Phần dư
axes[0].scatter(y_pred_best, residuals, alpha=0.6, color='blue')
axes[0].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[0].axhline(y=best_results['RMSE'], color='orange', linestyle=':', linewidth=1.5)
axes[0].axhline(y=-best_results['RMSE'], color='orange', linestyle=':', linewidth=1.5)
axes[0].set_title(f'Phân tích Phần dư ({best_model_name})')
axes[0].set_xlabel('Giá trị dự báo (°C)')
axes[0].set_ylabel('Phần dư (°C)')
axes[0].grid(alpha=0.3)

# Scatter: Thực tế vs Dự báo
scatter = axes[1].scatter(best_results['y_test'], y_pred_best, alpha=0.6, color='blue')
axes[1].plot([best_results['y_test'].min(), best_results['y_test'].max()], 
             [best_results['y_test'].min(), best_results['y_test'].max()], 
             'r--', lw=2, label='Đường hoàn hảo')
axes[1].set_title(f'Thực tế vs Dự báo ({best_model_name})')
axes[1].set_xlabel('Giá trị thực tế (°C)')
axes[1].set_ylabel('Giá trị dự báo (°C)')
axes[1].legend()
axes[1].grid(alpha=0.3)

# Thêm hiệu suất vào biểu đồ
mae_text = f'MAE: {best_results["MAE"]:.3f}°C'
rmse_text = f'RMSE: {best_results["RMSE"]:.3f}°C'
r2_text = f'R²: {best_results["R2"]:.3f}'
axes[1].text(0.05, 0.95, f'{mae_text}\n{rmse_text}\n{r2_text}', 
             transform=axes[1].transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.show()

# Biểu đồ 3: Phân phối phần dư
plt.figure(figsize=(10, 5))
n, bins, patches = plt.hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='lightcoral')
plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Mean=0')
plt.axvline(x=best_results['MAE'], color='blue', linestyle=':', linewidth=2, label=f'MAE={best_results["MAE"]:.2f}')
plt.axvline(x=-best_results['MAE'], color='blue', linestyle=':', linewidth=2)
plt.title('Phân phối Phần dư')
plt.xlabel('Phần dư (°C)')
plt.ylabel('Tần suất')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Biểu đồ 4: Feature importance (nếu là Random Forest)
if best_model_name == "Random Forest":
    print("\n  → Phân tích Feature Importance (Random Forest)...")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n    Top 10 feature quan trọng nhất:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"      {row['feature']:20}: {row['importance']:.4f}")
    
    plt.figure(figsize=(10, 6))
    bars = plt.barh(feature_importance['feature'][:15], feature_importance['importance'][:15], color='skyblue')
    plt.xlabel('Độ quan trọng')
    plt.title('Top 15 Feature Importance (Random Forest)')
    plt.gca().invert_yaxis()
    plt.grid(alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()

# Bước 6: Lưu mô hình tốt nhất
print(f"\n[6/6] Lưu mô hình...")
os.makedirs(MODEL_DIR, exist_ok=True)

# Tạo tên file an toàn
model_name_safe = best_model_name.lower().replace(' ', '_').replace('-', '_')
model_file = os.path.join(MODEL_DIR, f"model_{model_name_safe}.pkl")

with open(model_file, 'wb') as f:
    pickle.dump(best_model, f)

print(f"    Lưu mô hình: {model_file}")
print(f"    Kích thước dữ liệu train: {X_train.shape}")
print(f"    Kích thước dữ liệu test: {X_test.shape}")

print("\nHUẤN LUYỆN HOÀN TẤT")
print(f"  Mô hình tốt nhất: {best_model_name}")
print(f"  RMSE: {best_results['RMSE']:.4f} | MAE: {best_results['MAE']:.4f} | R²: {best_results['R2']:.4f}")
print(f"  File: {model_file}")
