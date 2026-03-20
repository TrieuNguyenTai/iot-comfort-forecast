# 🌡️ IoT Indoor Comfort Index Forecasting System
> An IoT system for predicting indoor comfort index (THI) in Cau Giay District, Hanoi (2022–2025)

![Python](https://img.shields.io/badge/Python-3.10-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange)
![IoT](https://img.shields.io/badge/IoT-ThingSpeak-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Overview

This system collects indoor temperature/humidity data from IoT sensors (ESP8266 + DHT11/22) combined with outdoor weather data from the **Open-Meteo API**, then trains Machine Learning models to **forecast the THI (Temperature Humidity Index)** and automatically recommend device control actions.

### Best Model Results (Random Forest)
| Metric | Value |
|--------|-------|
| R²     | 0.9500 |
| MAE    | 0.0133°C |
| RMSE   | 0.0304°C |

---

## 🗂️ Project Structure

```
.
├── 1_thu_thap_du_lieu.py   # Data collection from Open-Meteo & ThingSpeak
├── 2_tien_xu_ly.py         # Preprocessing, EDA, normalization
├── 3_huan_luyen.py         # Training Linear Regression & Random Forest
├── 4_giao_dien.py          # Tkinter GUI - real-time forecast display
├── requirements.txt
├── Dockerfile
├── .gitignore
└── README.md
```

---

## ⚙️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Hardware | ESP8266 + DHT11/DHT22 |
| IoT Platform | ThingSpeak |
| Weather API | Open-Meteo Archive API |
| ML Models | Linear Regression, Random Forest |
| GUI | Tkinter + Matplotlib |
| Language | Python 3.10 |

---

## 🚀 How to Run

### 1. Clone repo
```bash
git clone https://github.com/<your-username>/iot-comfort-forecast.git
cd iot-comfort-forecast
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run pipeline (in order)
```bash
# Step 1: Collect data
python 1_thu_thap_du_lieu.py

# Step 2: Preprocess
python 2_tien_xu_ly.py

# Step 3: Train models
python 3_huan_luyen.py

# Step 4: Launch GUI
python 4_giao_dien.py
```

### 4. Or run with Docker
```bash
docker build -t iot-comfort .
docker run iot-comfort
```

---

## 📊 Dataset

- **Outdoor:** [Open-Meteo Archive API](https://open-meteo.com/) — Cau Giay, Hanoi (21.0285°N, 105.8581°E)
- **Indoor:** ThingSpeak (DHT11/22 sensor via ESP8266)
- **Period:** 2022–2025
- **Size:** ~35,000 hourly data points
- **Download:** [Google Drive](https://drive.google.com/your-link-here) ← *replace with actual link*

---

## 🌡️ THI Comfort Levels

| THI (°C) | Level | Recommendation |
|----------|-------|----------------|
| < 20 | Very Cold | Turn on heater |
| 20–26 | Comfortable | Turn off devices |
| 26–30 | Slightly Hot | Turn on fan |
| 30–35 | Hot | AC at 26°C |
| > 35 | Very Hot | AC at 24°C |

---

## 📷 Demo

> *(Add GUI screenshot or demo video here)*

---

## 👤 Author

**Trieu Nguyen Tai** — Hanoi University of Mining and Geology, Faculty of IT