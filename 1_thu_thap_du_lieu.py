import requests
import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

THINGSPEAK_CHANNEL_ID = '2882365'
THINGSPEAK_API_KEY = '3FOBKUT3V4U3YVH8'
LAT, LON = 21.0285, 105.8581

# Bước 1: Lấy dữ liệu ngoài trời từ Open-Meteo (2022-2025)
def get_outdoor_data():
    print("\n[1/4] Lấy dữ liệu ngoài trời...")
    dfs = []
    
    for year in [2022, 2023, 2024]:
        try:
            url = "https://archive-api.open-meteo.com/v1/archive"
            params = {
                'latitude': LAT, 'longitude': LON,
                'start_date': f'{year}-01-01', 'end_date': f'{year}-12-31',
                'hourly': 'temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m,shortwave_radiation,cloud_cover,precipitation',
                'timezone': 'Asia/Bangkok'
            }
            data = requests.get(url, params=params, timeout=30).json()['hourly']
            df = pd.DataFrame({
                'thời_gian': pd.to_datetime(data['time']),
                'nhiệt_độ_ngoài': data['temperature_2m'],
                'độ_ẩm_ngoài': data['relative_humidity_2m'],
                'áp_suất': data['pressure_msl'],
                'tốc_độ_gió': data['wind_speed_10m'],
                'bức_xạ_mặt_trời': data['shortwave_radiation'],
                'độ_che_phủ_mây': data['cloud_cover'],
                'lượng_mưa': data['precipitation']
            })
            dfs.append(df)
            print(f"   Năm {year}: {len(df):,} điểm")
        except Exception as e:
            print(f"   Năm {year}: Lỗi - {str(e)[:50]}...")
        time.sleep(1)
    
    try:
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            'latitude': LAT, 'longitude': LON,
            'start_date': '2025-01-01', 'end_date': '2025-12-04',
            'hourly': 'temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m,shortwave_radiation,cloud_cover,precipitation',
            'timezone': 'Asia/Bangkok'
        }
        data = requests.get(url, params=params, timeout=30).json()['hourly']
        df = pd.DataFrame({
            'thời_gian': pd.to_datetime(data['time']),
            'nhiệt_độ_ngoài': data['temperature_2m'],
            'độ_ẩm_ngoài': data['relative_humidity_2m'],
            'áp_suất': data['pressure_msl'],
            'tốc_độ_gió': data['wind_speed_10m'],
            'bức_xạ_mặt_trời': data['shortwave_radiation'],
            'độ_che_phủ_mây': data['cloud_cover'],
            'lượng_mưa': data['precipitation']
        })
        dfs.append(df)
        print(f"   Năm 2025: {len(df):,} điểm")
    except Exception as e:
        print(f"   Năm 2025: Lỗi - {str(e)[:50]}...")
    
    outdoor_df = pd.concat(dfs, ignore_index=True).sort_values('thời_gian').reset_index(drop=True)
    print(f"  → Tổng outdoor: {len(outdoor_df):,} điểm\n")
    return outdoor_df

# Bước 2: Lấy dữ liệu ThingSpeak (3000 điểm cuối)
def get_thingspeak_data():
    print("[2/4] Lấy dữ liệu ThingSpeak...")
    try:
        url = f"https://api.thingspeak.com/channels/{THINGSPEAK_CHANNEL_ID}/feeds.json"
        data = requests.get(url, params={'api_key': THINGSPEAK_API_KEY, 'results': 3000}, timeout=30).json()
        
        indoor_data = []
        for feed in data.get('feeds', []):
            try:
                temp, humidity = float(feed['field1']), float(feed['field2'])
                indoor_data.append({
                    'thời_gian': pd.to_datetime(feed['created_at']).tz_localize(None),
                    'nhiệt_độ_trong_thực': temp,
                    'độ_ẩm_trong_thực': humidity
                })
            except:
                pass
        
        if indoor_data:
            indoor_df = pd.DataFrame(indoor_data).sort_values('thời_gian').reset_index(drop=True)
            print(f"   ThingSpeak: {len(indoor_df):,} điểm\n")
            return indoor_df
        else:
            print(f"   ThingSpeak: Không có dữ liệu\n")
            return None
    except Exception as e:
        print(f"   ThingSpeak: Lỗi - {str(e)[:50]}...\n")
        return None

# Bước 3: Sinh dữ liệu trong nhà từ dữ liệu ngoài trời
def generate_indoor_data(outdoor_df):
    print("[3/4] Sinh dữ liệu trong nhà...")
    df = outdoor_df.copy()
    
    # Xử lý missing values
    for col in ['nhiệt_độ_ngoài', 'độ_ẩm_ngoài', 'lượng_mưa']:
        df[col] = df[col].fillna(df[col].mean())
    
    # Xử lý outliers
    df['nhiệt_độ_ngoài'] = df['nhiệt_độ_ngoài'].clip(-10, 50)
    df['độ_ẩm_ngoài'] = df['độ_ẩm_ngoài'].clip(0, 100)
    df['lượng_mưa'] = df['lượng_mưa'].clip(0, 500)
    
    # Sinh nhiệt độ trong nhà
    rolling_mean = df['nhiệt_độ_ngoài'].rolling(window=3, center=True, min_periods=1).mean()
    df['nhiệt_độ_trong'] = (
        df['nhiệt_độ_ngoài'] * 0.5 + rolling_mean * 0.3 +
        np.random.normal(0, 0.25, len(df)) + 2
    ).clip(16, 34).round(1)
    
    # Sinh độ ẩm trong nhà
    rolling_rain = df['lượng_mưa'].rolling(window=6, center=True, min_periods=1).mean()
    df['độ_ẩm_trong'] = (
        df['độ_ẩm_ngoài'] * 0.65 + rolling_rain * 1.5 +
        np.random.normal(0, 1.5, len(df)) + 8
    ).clip(32, 92).round(1)
    
    print(f"   Synthetic: {len(df):,} điểm\n")
    return df[['thời_gian', 'nhiệt_độ_trong', 'độ_ẩm_trong']]

def calculate_thi_celsius(temperature_c, humidity):
    """
    Tính chỉ số THI (Temperature-Humidity Index) cho độ Celsius
    Sử dụng công thức đơn giản: THI = T - (0.55 - 0.0055*RH) * (T - 14.5)
    Trong đó:
        T = nhiệt độ (°C)
        RH = độ ẩm tương đối (%)
    """
    thi = temperature_c - (0.55 - 0.0055 * humidity) * (temperature_c - 14.5)
    return thi

# Bước 4: Hợp nhất dữ liệu + tính toán thuộc tính
def process_data(outdoor_df, thingspeak_df, indoor_synthetic_df):
    print("[4/4] Xử lý và tính toán...")
    
    # Hợp nhất outdoor + synthetic indoor
    df = outdoor_df.merge(indoor_synthetic_df, on='thời_gian', how='left')
    print(f"  → Sau hợp nhất: {len(df):,} điểm")
    
    # Thay thế dữ liệu thực từ ThingSpeak
    if thingspeak_df is not None and len(thingspeak_df) > 0:
        thingspeak_df.columns = ['thời_gian', 'nhiệt_độ_trong_thực', 'độ_ẩm_trong_thực']
        df = pd.merge_asof(df, thingspeak_df, on='thời_gian', direction='nearest', tolerance=pd.Timedelta('1H'))
        mask = df['nhiệt_độ_trong_thực'].notna()
        df.loc[mask, 'nhiệt_độ_trong'] = df.loc[mask, 'nhiệt_độ_trong_thực']
        df.loc[mask, 'độ_ẩm_trong'] = df.loc[mask, 'độ_ẩm_trong_thực']
        df = df.drop(['nhiệt_độ_trong_thực', 'độ_ẩm_trong_thực'], axis=1)
        print(f"  → Thay {mask.sum():,} điểm bằng dữ liệu thực")
    
    # Tính toán time features
    df['giờ'] = df['thời_gian'].dt.hour
    df['ngày'] = df['thời_gian'].dt.day
    df['tháng'] = df['thời_gian'].dt.month
    df['thứ'] = df['thời_gian'].dt.day_name()
    df['cuối_tuần'] = (df['thời_gian'].dt.dayofweek >= 5).astype(int)
    
    # Xác định 4 mùa
    def get_season(month):
        if month in [3, 4, 5]: return 'Xuân'
        elif month in [6, 7, 8]: return 'Hè'
        elif month in [9, 10, 11]: return 'Thu'
        else: return 'Đông'
    df['mùa'] = df['tháng'].apply(get_season)
    
    # Tính chênh lệch
    df['chênh_lệch_nhiệt_độ'] = (df['nhiệt_độ_trong'] - df['nhiệt_độ_ngoài']).round(1)
    df['chênh_lệch_độ_ẩm'] = (df['độ_ẩm_trong'] - df['độ_ẩm_ngoài']).round(1)
    
    # Tính điểm sương trong nhà
    T = df['nhiệt_độ_trong']
    RH = df['độ_ẩm_trong']
    df['điểm_sương_trong'] = (
        243.5 * np.log(RH/100 + 1e-10) + 17.27 * T / (243.5 + T)
    ) / (17.27 - np.log(RH/100 + 1e-10) - 17.27 * T / (243.5 + T))
    df['điểm_sương_trong'] = df['điểm_sương_trong'].round(1)
    
    # TÍNH CHỈ SỐ THOẢI MÁI (THI) - SỬA LẠI CÔNG THỨC
    print("  → Tính chỉ số thoải mái (THI)...")
    df['chỉ_số_thoải_mái'] = calculate_thi_celsius(df['nhiệt_độ_trong'], df['độ_ẩm_trong'])
    df['chỉ_số_thoải_mái'] = df['chỉ_số_thoải_mái'].round(1)
    
    # Chọn cột cuối cùng
    cols = ['thời_gian', 'nhiệt_độ_ngoài', 'độ_ẩm_ngoài', 'áp_suất', 'tốc_độ_gió',
            'bức_xạ_mặt_trời', 'độ_che_phủ_mây', 'lượng_mưa', 'nhiệt_độ_trong', 'độ_ẩm_trong',
            'giờ', 'ngày', 'tháng', 'thứ', 'mùa', 'cuối_tuần',
            'chênh_lệch_nhiệt_độ', 'chênh_lệch_độ_ẩm', 'điểm_sương_trong', 'chỉ_số_thoải_mái']
    
    df = df[cols].dropna().drop_duplicates(subset=['thời_gian']).sort_values('thời_gian').reset_index(drop=True)
    print(f"  → Sau xử lý: {len(df):,} điểm\n")
    
    # Kiểm tra giá trị THI
    print("  → Kiểm tra chỉ số THI:")
    print(f"    Min THI: {df['chỉ_số_thoải_mái'].min():.1f}°C")
    print(f"    Max THI: {df['chỉ_số_thoải_mái'].max():.1f}°C")
    print(f"    Mean THI: {df['chỉ_số_thoải_mái'].mean():.1f}°C")
    
    # Kiểm tra phân phối THI
    thi_counts = pd.cut(df['chỉ_số_thoải_mái'], 
                       bins=[-np.inf, 20, 26, 30, 35, np.inf],
                       labels=['Rất lạnh', 'Thoải mái', 'Hơi nóng', 'Nóng', 'Rất nóng'])
    print("\n  → Phân phối mức độ thoải mái:")
    for level, count in thi_counts.value_counts().sort_index().items():
        percentage = count / len(df) * 100
        print(f"    {level}: {count:,} điểm ({percentage:.1f}%)")
    
    return df

# MAIN
if __name__ == '__main__':
    print("\nTHU THẬP DỮ LIỆU - HỆ THỐNG IoT DỰ BÁO CHỈ SỐ THOẢI MÁI")
    print(f"  Vị trí: Cầu Giấy, Hà Nội ({LAT}, {LON})")
    print(f"  Thời gian: 2022-2025")
    
    outdoor_df = get_outdoor_data()
    thingspeak_df = get_thingspeak_data()
    indoor_synthetic_df = generate_indoor_data(outdoor_df)
    final_df = process_data(outdoor_df, thingspeak_df, indoor_synthetic_df)
    
    final_df.to_csv('dữ_liệu_tổng.csv', index=False, encoding='utf-8-sig')
    
    print("\nTHU THẬP HOÀN TẤT")
    print(f"  Lưu thành công: dữ_liệu_tổng.csv")
    print(f"  Số lượng: {len(final_df):,} dòng, {len(final_df.columns)} cột")
    print(f"  Thời gian: {final_df['thời_gian'].min().date()} đến {final_df['thời_gian'].max().date()}")
    print(f"  Chỉ số THI: {final_df['chỉ_số_thoải_mái'].min():.1f}°C - {final_df['chỉ_số_thoải_mái'].max():.1f}°C")
