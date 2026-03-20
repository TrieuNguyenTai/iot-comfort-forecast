import tkinter as tk
from tkinter import ttk
import pandas as pd
import numpy as np
import pickle
import datetime
import requests
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings('ignore')

# CẤU HÌNH VÀ HẰNG SỐ
INPUT_FILE = 'dữ_liệu_sạch.csv'
SCALER_FILE = 'scaler.pkl'
MODEL_FILE = 'models/model_random_forest.pkl' 

THINGSPEAK_CHANNEL_ID = '2882365'
THINGSPEAK_API_KEY = '3FOBKUT3V4U3YVH8'

LAT, LON = 21.0285, 105.8581  # Quận Cầu Giấy, Hà Nội

SCALED_FEATURES = [
    'nhiệt_độ_ngoài', 'độ_ẩm_ngoài', 'áp_suất', 'tốc_độ_gió',
    'bức_xạ_mặt_trời', 'lượng_mưa',
    'nhiệt_độ_trong', 'độ_ẩm_trong',
    'giờ', 'tháng', 'mùa_mã', 'cuối_tuần',
]

# LOAD DỮ LIỆU VÀ MÔ HÌNH
try:
    df = pd.read_csv(INPUT_FILE, encoding='utf-8-sig')
    scaler = pickle.load(open(SCALER_FILE, 'rb'))
    model = pickle.load(open(MODEL_FILE, 'rb'))
    print("\nLoad dữ liệu, scaler và mô hình thành công")
except FileNotFoundError as e:
    print(f"Lỗi: {e}. Vui lòng kiểm tra file cấu hình.")
    # Đặt giá trị None nếu có lỗi để tránh NameError khi chạy GUI
    df, scaler, model = None, None, None 

# CÁC HÀM HỖ TRỢ
def classify_comfort(thi_value):
    """Phân loại mức độ thoải mái dựa trên Heat Index (THI)"""
    if thi_value < 20:
        return "RẤT LẠNH"
    elif thi_value < 26:
        return "THOẢI MÁI"
    elif thi_value < 30:
        return "HƠI NÓNG"
    elif thi_value < 35:
        return "NÓNG"
    else:
        return "RẤT NÓNG"

def get_device_action(thi_value):
    """Đề xuất hành động điều khiển thiết bị"""
    if thi_value < 20:
        return "BẬT SƯỞI & TẮT QUẠT (Tăng nhiệt độ)"
    elif thi_value < 26:
        return "DUY TRÌ/TẮT THIẾT BỊ (Tiết kiệm năng lượng)"
    elif thi_value < 30:
        return "BẬT QUẠT HOẶC Đ.H. CHẾ ĐỘ QUẠT (Làm mát nhẹ)"
    elif thi_value < 35:
        return "BẬT Đ.H. 26°C (Làm mát vừa)"
    else:
        return "BẬT Đ.H. 24°C & TẮT QUẠT (Làm mát mạnh)"

def get_marker_color(v, text=False):
    """Lấy mã màu (cho biểu đồ) hoặc tên màu chữ (cho nhãn) dựa trên giá trị THI"""
    if v < 20: return '#17A2B8' if not text else 'blue'
    elif 20 <= v < 26: return '#28A745' if not text else 'green'
    elif 26 <= v < 30: return '#FFC107' if not text else '#CC8800'
    else: return '#DC3545' if not text else 'red'

def get_thingspeak_data():
    """Lấy dữ liệu cảm biến TRONG NHÀ từ ThingSpeak"""
    try:
        url = f"https://api.thingspeak.com/channels/{THINGSPEAK_CHANNEL_ID}/feeds.json"
        params = {'api_key': THINGSPEAK_API_KEY, 'results': 1}
        response = requests.get(url, params=params, timeout=5).json()
        
        default_temp = 25.0
        default_humid = 53.9

        if 'feeds' in response and len(response['feeds']) > 0:
            data = response['feeds'][-1]
            temp_in = float(data.get('field1', default_temp))
            humid_in = float(data.get('field2', default_humid))
            
            if not (16 <= temp_in <= 34): temp_in = default_temp
            if not (30 <= humid_in <= 95): humid_in = default_humid
            
            return temp_in, humid_in
        return default_temp, default_humid
    except Exception:
        # Trả về giá trị mặc định nếu có lỗi mạng/API
        return 25.0, 53.9

def get_openmeteo_forecast():
    """Lấy dữ liệu DỰ BÁO (7 giờ tới) từ Open-Meteo"""
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            'latitude': LAT, 'longitude': LON,
            'hourly': 'temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m,shortwave_radiation,precipitation',
            'timezone': 'Asia/Bangkok', 'forecast_days': 1
        }
        response = requests.get(url, params=params, timeout=5).json()
        
        if 'hourly' in response:
            hourly = response['hourly']
            now = datetime.datetime.now()
            times = [datetime.datetime.fromisoformat(t.replace('Z', '+00:00')) for t in hourly['time']]
            
            current_hour_idx = -1
            for i, t in enumerate(times):
                if t.hour == now.hour and t.date() == now.date():
                    current_hour_idx = i
                    break
            if current_hour_idx == -1: current_hour_idx = 0
            
            forecast_data = []
            for i in range(7): 
                idx = current_hour_idx + i
                if idx < len(hourly['time']):
                    forecast_time = times[idx]
                    forecast_data.append({
                        'hour_of_day': forecast_time.hour,
                        'date_time': forecast_time,
                        'temp_out': hourly['temperature_2m'][idx],
                        'humid_out': hourly['relative_humidity_2m'][idx],
                        'pressure': hourly['pressure_msl'][idx] if hourly['pressure_msl'] else 1013.0,
                        'wind_speed': hourly['wind_speed_10m'][idx] if hourly['wind_speed_10m'] else 2.0,
                        'radiation': hourly['shortwave_radiation'][idx] if hourly['shortwave_radiation'] else 0.0,
                        'rainfall': hourly['precipitation'][idx] if hourly['precipitation'] else 0.0,
                    })
            return forecast_data
        return None
    except Exception:
        return None

def get_time_features(dt_obj):
    """Tạo các feature thời gian và mùa/cuối tuần"""
    month = dt_obj.month
    dayofweek = dt_obj.weekday()
    
    def get_season_code(m):
        if m in [3, 4, 5]: return 3  
        elif m in [6, 7, 8]: return 2  
        elif m in [9, 10, 11]: return 1  
        else: return 0  
        
    return {
        'giờ': dt_obj.hour,
        'tháng': month,
        'mùa_mã': get_season_code(month),
        'cuối_tuần': 1 if dayofweek >= 5 else 0
    }

def predict_thi_full(data_row, dt_obj):
    """Dự báo chỉ số thoải mái"""
    if scaler is None or model is None:
        return 25.0
        
    time_features = get_time_features(dt_obj)
    
    row_input_dict = {
        'nhiệt_độ_ngoài': [data_row['temp_out']], 'độ_ẩm_ngoài': [data_row['humid_out']], 
        'áp_suất': [data_row.get('pressure', 1013.0)], 'tốc_độ_gió': [data_row.get('wind_speed', 2.0)],
        'bức_xạ_mặt_trời': [data_row.get('radiation', 0.0)], 'lượng_mưa': [data_row.get('rainfall', 0.0)],
        'nhiệt_độ_trong': [data_row['temp_in']], 'độ_ẩm_trong': [data_row['humid_in']],
        'giờ': [time_features['giờ']], 'tháng': [time_features['tháng']],
        'mùa_mã': [time_features['mùa_mã']], 'cuối_tuần': [time_features['cuối_tuần']]
    }
    
    row = pd.DataFrame(row_input_dict)
    row = row[SCALED_FEATURES] 

    try:
        row_scaled = scaler.transform(row.values)
        thi = model.predict(row_scaled)[0]
    except Exception:
        return 25.0
        
    return thi

def predict_comfort():
    """Hàm chính thực hiện dự báo và cập nhật GUI"""
    
    # 1. Lấy dữ liệu
    temp_in_current, humid_in_current = get_thingspeak_data()
    forecast_data = get_openmeteo_forecast()
    
    # Xóa cảnh báo lỗi cũ (nếu có)
    for widget in frame_status.winfo_children():
        if widget.winfo_name() == 'error_label':
            widget.destroy()

    if forecast_data is None or model is None:
        # Nếu có lỗi kết nối hoặc lỗi load mô hình
        frame_status.config(bg="#FFDDEE")
        
        # Ẩn các nhãn thông thường
        lbl_status_main.grid_forget()
        lbl_temp_in_label.grid_forget()
        lbl_temp_in.grid_forget()
        lbl_temp_out_label.grid_forget()
        lbl_temp_out.grid_forget()
        lbl_thi_label.grid_forget()
        lbl_thi.grid_forget()
        
        # Hiện thông báo lỗi
        error_text = "KHÔNG THỂ KẾT NỐI DỮ LIỆU NGOÀI TRỜI (Open-Meteo) HOẶC LỖI LOAD MÔ HÌNH."
        lbl_error = tk.Label(frame_status, text=error_text, 
                             fg="red", bg="#FFDDEE", font=("Arial", 14, "bold"), padx=20, pady=18, name='error_label')
        lbl_error.grid(row=0, column=0, columnspan=2, sticky='nsew') 
        
        lbl_alert.config(text="HỆ THỐNG GẶP LỖI. KHÔNG THỂ DỰ BÁO.", fg="#DC3545", bg="#FFF0E0")
        
        # Xóa biểu đồ cũ
        ax.clear()
        ax.set_title("LỖI DỮ LIỆU", fontsize=16, fontweight='bold', color='red')
        canvas.draw()
        
        return
        
    # Đảm bảo các nhãn thông thường được hiển thị (trường hợp phục hồi sau lỗi)
    lbl_status_main.grid(row=0, column=0, columnspan=2, sticky='w', padx=20, pady=5)
    lbl_temp_in_label.grid(row=1, column=0, sticky='w', pady=2)
    lbl_temp_in.grid(row=1, column=1, sticky='w', pady=2)
    lbl_temp_out_label.grid(row=2, column=0, sticky='w', pady=2)
    lbl_temp_out.grid(row=2, column=1, sticky='w', pady=2)
    lbl_thi_label.grid(row=3, column=0, sticky='w', pady=(10, 5))
    lbl_thi.grid(row=3, column=1, sticky='w', pady=(10, 5))

    
    now = datetime.datetime.now()
    forecast_current = forecast_data[0]
    
    current_data_row = {**forecast_current, 'temp_in': temp_in_current, 'humid_in': humid_in_current}
    
    # 2. Dự báo chỉ số hiện tại
    thi_current = predict_thi_full(current_data_row, now)
    comfort_level = classify_comfort(thi_current)
    
    # Cập nhật trạng thái chính
    bg_color_status = "#E6F7FF"
    if "THOẢI MÁI" in comfort_level: bg_color_status = "#E6FFEE"
    elif "NÓNG" in comfort_level or "LẠNH" in comfort_level: bg_color_status = "#FFF3E0"
    
    frame_status.config(bg=bg_color_status)
    lbl_status_main.config(bg=bg_color_status)
    lbl_status_main.config(text=f"Vị trí: CẦU GIẤY, HÀ NỘI | {now.strftime('%H:%M:%S - %d/%m/%Y')}", fg="black", font=("Arial", 14, "bold"), pady=10)

    # Cập nhật thông số chi tiết
    lbl_temp_in.config(text=f"{temp_in_current:.1f}°C / {humid_in_current:.1f}% (IoT)", bg=bg_color_status)
    lbl_temp_out.config(text=f"{forecast_current['temp_out']:.1f}°C / {forecast_current['humid_out']:.0f}% (Dự báo)", bg=bg_color_status)
    lbl_thi.config(text=f"{thi_current:.1f}°C ({comfort_level})", bg=bg_color_status, fg=get_marker_color(thi_current, text=True))
    
    # 3. Dự báo Tương lai
    plot_data = [{'time': "Hiện tại", 'thi': thi_current, 'dt': now}]
    
    forecast_indices = [(1, 1), (3, 3), (6, 6)]
    
    for hours, idx in forecast_indices:
        if idx < len(forecast_data):
            forecast = forecast_data[idx]
            future_dt = forecast['date_time'] 

            # Ước tính nhiệt độ/độ ẩm trong nhà cho tương lai (đơn giản hóa)
            temp_in_pred = np.clip(forecast['temp_out'] * 0.5 + temp_in_current * 0.4 + 2, 16, 34)
            humid_in_pred = np.clip(forecast['humid_out'] * 0.65 + 8, 32, 92)
            
            future_data_row = {**forecast, 'temp_in': temp_in_pred, 'humid_in': humid_in_pred}
            thi_pred_raw = predict_thi_full(future_data_row, future_dt)
            
            # Làm mịn/hạn chế thay đổi quá nhanh
            max_change_per_hour = 0.25
            max_allowed_change = max_change_per_hour * hours + 1.0 
            
            if abs(thi_pred_raw - thi_current) > max_allowed_change:
                thi = thi_current + max_allowed_change if thi_pred_raw > thi_current else thi_current - max_allowed_change
            else:
                thi = thi_pred_raw
            
            thi = round(thi, 1)
            
            plot_data.append({
                'time': f"+{hours}h ({future_dt.hour}h)", 
                'thi': thi, 
                'dt': future_dt,
                'level': classify_comfort(thi),
                'action': get_device_action(thi)
            })
    
    # 4. Cập nhật bảng
    for item in tree.get_children(): tree.delete(item)
    
    tree.insert('', 'end', values=(
            "Hiện tại", "IoT/Dự báo", f"{thi_current:.1f}°C",
            classify_comfort(thi_current), get_device_action(thi_current)
        ), tags=('current',))

    for pred in plot_data[1:]:
        tree.insert('', 'end', values=(
            pred['time'], "Model", f"{pred['thi']:.1f}°C",
            pred['level'], pred['action']
        ))
    
    tree.tag_configure('current', background='#DDEEFF', font=('Arial', 13, 'bold'))

    # 5. Vẽ biểu đồ
    ax.clear()
    plot_times = [p['time'] for p in plot_data]
    plot_thi_values = [p['thi'] for p in plot_data]
    
    colors = [get_marker_color(v) for v in plot_thi_values]
    
    ax.plot(plot_times, plot_thi_values, marker='o', linewidth=4, markersize=10, color='#004D99', linestyle='-', zorder=2)
    ax.scatter(plot_times, plot_thi_values, c=colors, s=250, zorder=5, alpha=0.9, edgecolors='black', linewidth=1.5)
    
    ax.axhspan(20, 26, color='#28A745', alpha=0.20, label='Thoải mái (20-26°C)')
    ax.axhspan(26, 30, color='#FFC107', alpha=0.15, label='Hơi nóng (26-30°C)')
    ax.axhspan(0, 20, color='#17A2B8', alpha=0.10)
    ax.axhspan(30, 40, color='#DC3545', alpha=0.15)

    ax.set_ylabel("Chỉ số thoải mái (°C)", fontsize=14, fontweight='bold', color='#333333')
    ax.set_xlabel("Thời gian", fontsize=14, fontweight='bold', color='#333333')
    ax.set_title("DỰ BÁO CHỈ SỐ THOẢI MÁI (HIỆN TẠI VÀ TƯƠNG LAI)", fontsize=16, fontweight='bold', color='#004D99')
    
    ax.grid(True, alpha=0.5, linewidth=1.0, linestyle=':', color='gray') 
    
    y_min, y_max = min(plot_thi_values) - 2, max(plot_thi_values) + 2
    ax.set_ylim([max(16, y_min), min(40, y_max)])
    ax.tick_params(axis='both', labelsize=12)
    
    for i, v in enumerate(plot_thi_values):
        ax.text(i, v + 0.9, f"{v:.1f}", ha='center', fontsize=12, fontweight='bold', color='black')
        
    ax.legend(loc='upper left', fontsize=10)
    canvas.draw()
    
    # 6. Cảnh báo
    alerts = []
    all_predictions = plot_data
    for p in all_predictions:
        time_label = p['time']
        thi = p['thi']
        action = get_device_action(thi)
        
        if thi < 20:
            alerts.append(f"{time_label}: THI {thi:.1f}°C -> RẤT LẠNH! Khuyến nghị: {action}")
        elif thi >= 30:
            alerts.append(f"{time_label}: THI {thi:.1f}°C -> NGUY HIỂM NÓNG! Khuyến nghị: {action}")
        elif 26 <= thi < 30:
            alerts.append(f"{time_label}: THI {thi:.1f}°C -> HƠI NÓNG. Khuyến nghị: {action}")
            
    alert_text = "\n".join(alerts) if alerts else "ĐIỀU KIỆN DỰ BÁO BÌNH THƯỜNG TRONG 6H TỚI"
    
    alert_fg_color = "#004D99"
    alert_bg_color = "#F7F7F7"
    if any("NGUY HIỂM NÓNG" in a for a in alerts) or any("RẤT LẠNH" in a for a in alerts):
        alert_fg_color = "#DC3545"
        alert_bg_color = "#FFF0E0"
    elif alerts:
        alert_fg_color = "#FFC107"
        alert_bg_color = "#FFFFE0"

    lbl_alert.config(text=alert_text, fg=alert_fg_color, bg=alert_bg_color)

# GIAO DIỆN (GUI)
root = tk.Tk()
root.title("HỆ THỐNG IoT DỰ BÁO CHỈ SỐ THOẢI MÁI (Cầu Giấy, Hà Nội)")
root.geometry("1200x1000")
root.resizable(True, True)
root.config(bg="#F8F9FA")

style = ttk.Style()
style.theme_use('clam') 
style.configure('Treeview', font=('Arial', 13), rowheight=35, background="#FFFFFF", fieldbackground="#FFFFFF", foreground="#333333")
style.configure('Treeview.Heading', font=('Arial', 13, 'bold'), background="#007BFF", foreground="white") 
style.map('Treeview', background=[('selected', '#007BFF')])

# Tiêu đề
title = tk.Label(
    root,
    text="HỆ THỐNG IoT DỰ BÁO CHỈ SỐ THOẢI MÁI TRONG NHÀ", 
    font=("Arial", 22, "bold"),
    bg="#004D99", 
    fg="white",
    pady=18
)
title.pack(fill=tk.X)

# Nút cập nhật
btn_update = tk.Button(
    root,
    text="CẬP NHẬT DỮ LIỆU VÀ DỰ BÁO",
    command=predict_comfort,
    font=("Arial", 16, "bold"),
    bg="#00BFFF", 
    fg="white",
    padx=40,
    pady=15,
    relief=tk.RAISED
)
btn_update.pack(pady=20)

# Nhãn trạng thái (Frame chỉ vừa vặn nội dung và căn giữa)
frame_status = tk.Frame(root, bd=2, relief=tk.RIDGE, bg="#E6F7FF")
# Bỏ fill=tk.X để frame chỉ vừa nội dung và được căn giữa theo pack mặc định
frame_status.pack(pady=10, padx=30) 

# Cấu hình Grid bên trong frame_status (Chỉ dùng 2 cột nội dung: 0 và 1)

# Nhãn trạng thái chính (Vị trí, Thời gian)
lbl_status_main = tk.Label(
    frame_status,
    text="Đang khởi động...",
    font=("Arial", 14, "bold"),
    justify=tk.LEFT,
    wraplength=1100,
    padx=20,
    pady=5,
    bg="#E6F7FF"
)
# Đặt vào cột 0, span 2 cột
lbl_status_main.grid(row=0, column=0, columnspan=2, sticky='w', padx=20, pady=5) 

# === Thông số chi tiết ===
# Hàng 1: TRONG NHÀ
lbl_temp_in_label = tk.Label(frame_status, text="TRONG NHÀ (T°C / %RH):", font=("Arial", 13, "bold"), anchor='w', padx=20, bg="#E6F7FF")
lbl_temp_in_label.grid(row=1, column=0, sticky='w', pady=2)
lbl_temp_in = tk.Label(frame_status, text="--°C / --% (IoT)", font=("Arial", 13), anchor='w', bg="#E6F7FF")
lbl_temp_in.grid(row=1, column=1, sticky='w', pady=2)

# Hàng 2: NGOÀI TRỜI
lbl_temp_out_label = tk.Label(frame_status, text="NGOÀI TRỜI (T°C / %RH):", font=("Arial", 13, "bold"), anchor='w', padx=20, bg="#E6F7FF")
lbl_temp_out_label.grid(row=2, column=0, sticky='w', pady=2)
lbl_temp_out = tk.Label(frame_status, text="--°C / --% (Dự báo)", font=("Arial", 13), anchor='w', bg="#E6F7FF")
lbl_temp_out.grid(row=2, column=1, sticky='w', pady=2)

# Hàng 3: Chỉ số THI
lbl_thi_label = tk.Label(frame_status, text="CHỈ SỐ THOẢI MÁI (THI):", font=("Arial", 14, "bold"), anchor='w', padx=20, pady=10, bg="#E6F7FF")
lbl_thi_label.grid(row=3, column=0, sticky='w', pady=(10, 5))
lbl_thi = tk.Label(frame_status, text="--°C (--)", font=("Arial", 14, "bold"), fg="black", anchor='w', bg="#E6F7FF")
lbl_thi.grid(row=3, column=1, sticky='w', pady=(10, 5))


# Bảng dự báo
lbl_table = tk.Label(root, text="BẢNG DỰ BÁO (HIỆN TẠI VÀ TƯƠNG LAI)", font=("Arial", 16, "bold"), bg="#F8F9FA", fg="#004D99")
lbl_table.pack(pady=(15, 10))

frame_tree = tk.Frame(root)
frame_tree.pack(pady=5, padx=30, fill=tk.BOTH, expand=False)

tree = ttk.Treeview(
    frame_tree,
    columns=('Time', 'Source', 'THI', 'Level', 'Action'),
    height=4,
    show='headings'
)

tree.column('Time', width=150, anchor='center')
tree.column('Source', width=120, anchor='center')
tree.column('THI', width=100, anchor='center')
tree.column('Level', width=220, anchor='center')
tree.column('Action', width=450, anchor='center')

tree.heading('Time', text='Thời điểm')
tree.heading('Source', text='Nguồn')
tree.heading('THI', text='Chỉ số (°C)')
tree.heading('Level', text='Mức độ Thoải mái')
tree.heading('Action', text='Khuyến nghị Điều khiển Thiết bị')

tree.pack(fill=tk.BOTH, expand=True)

# Nhãn cảnh báo
frame_alert = tk.Frame(root, bd=2, relief=tk.SUNKEN, bg="#FFF0E0")
frame_alert.pack(pady=15, padx=30, fill=tk.X)

lbl_alert = tk.Label(
    frame_alert,
    text="",
    font=("Arial", 13, "bold"),
    justify=tk.LEFT,
    wraplength=1100,
    bg="#FFF0E0",
    padx=20,
    pady=15
)
lbl_alert.pack(fill=tk.X)

# Biểu đồ
fig, ax = plt.subplots(figsize=(10, 4.5), dpi=100)
fig.patch.set_facecolor('#F8F9FA') 
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.get_tk_widget().pack(pady=15, padx=30, fill=tk.BOTH, expand=True)

# Chân trang
footer = tk.Label(
    root,
    text="Dữ liệu: ThingSpeak (IoT) + Open-Meteo | Mô hình: Random Forest Regression",
    font=("Arial", 11),
    fg="gray",
    bg="#F8F9FA",
    pady=12
)
footer.pack(fill=tk.X, side=tk.BOTTOM)

# Dự báo lần đầu
# Cần dùng root.after(0, ...) để đảm bảo các widget đã được tạo trước khi gọi predict_comfort
if model is not None:
    root.after(100, predict_comfort)

root.mainloop()