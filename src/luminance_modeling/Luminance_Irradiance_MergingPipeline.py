import numpy as np
import cv2
import os
import glob
import pandas as pd
from pvlib.solarposition import get_solarposition
from datetime import datetime
import pytz

# === Load solarimetric station data (June–October 2015) ===

excel_path = r"C:\Users\artur\Documents\ASI Solar Analytics\spreadsheets\SolarStation_Data_2015_Jun_Oct.xlsx"
sheets = ["07-15", "08-15", "09-15", "10-15"]
columns = [0] + list(range(14, 26))  # Column A + columns O to Z

dfs = []
for sheet in sheets:
    df = pd.read_excel(excel_path, sheet_name=sheet, usecols=columns, skiprows=4, header=None)
    df.columns = ['datetime', 'Global_Avg', 'Global_Std', 'Global_Max', 'Global_Min',
                  'Diffuse_Avg', 'Diffuse_Std', 'Diffuse_Max', 'Diffuse_Min',
                  'Direct_Avg', 'Direct_Std', 'Direct_Max', 'Direct_Min']
    df['datetime'] = pd.to_datetime(df['datetime'], dayfirst=True, errors='coerce')
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    dfs.append(df)

df_station = pd.concat(dfs, ignore_index=True)
output_station_csv = r"C:\Users\artur\Documents\ASI Solar Analytics\spreadsheets\Processed_SolarStation_Data_2015.csv"
df_station.to_csv(output_station_csv, index=False, sep=';', float_format='%.10f', encoding='utf-8')

print("✅ Solar station CSV saved to:")
print(output_station_csv)

# === Image luminance estimation using hemispherical sampling ===

def detect_sun(image_rgb):
    red_channel = image_rgb[:, :, 0]
    _, thresh = cv2.threshold(red_channel, 240, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M["m00"] != 0:
            return np.array([int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])])
    return None

def solar_direction(cx, cy, shape, fov=180):
    h, w = shape[:2]
    f = w / (2 * np.sin(np.radians(fov / 2)))
    dx = (cx - w / 2)
    dy = (cy - h / 2)
    r = np.sqrt(dx**2 + dy**2)
    if r == 0:
        return np.array([0, 0, 1])
    theta = 2 * np.arcsin(r / (2 * f))
    return np.array([
        np.sin(theta) * dx / r,
        np.sin(theta) * dy / r,
        np.cos(theta)
    ])

def hemispherical_sampling(n=5000):
    R1, R2 = np.random.rand(n), np.random.rand(n)
    phi = 2 * np.pi * R1
    theta = np.arccos(np.sqrt(R2))
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.vstack((x, y, z)).T

def rotation_matrix(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    if s == 0:
        return np.eye(3)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat @ kmat * ((1 - c) / s**2)

def project_points(shape, points, fov=180):
    h, w = shape[:2]
    f = w / (2 * np.sin(np.radians(fov / 2)))
    x = (points[:, 0] / (1 + points[:, 2])) * f + w / 2
    y = (points[:, 1] / (1 + points[:, 2])) * f + h / 2
    return np.vstack([x, y]).T

def extract_rgb(image_rgb, coords):
    coords = np.round(coords).astype(int)
    coords[:, 0] = np.clip(coords[:, 0], 0, image_rgb.shape[1]-1)
    coords[:, 1] = np.clip(coords[:, 1], 0, image_rgb.shape[0]-1)
    return image_rgb[coords[:, 1], coords[:, 0], :]

def compute_luminance(rgb):
    Y = 0.2126 * rgb[:, 0] + 0.7152 * rgb[:, 1] + 0.0722 * rgb[:, 2]
    Y_corr = 255 * (Y / 255) ** 2.2
    return Y_corr.mean()

def extract_image_metadata(filename):
    parts = filename.replace(".jpg", "").split("_")
    return {
        "date": f"{parts[2]}-{parts[3]}-{parts[4]}",
        "time": f"{parts[5]}:{parts[6]}:{parts[7]}",
        "year": int(parts[2]),
        "month": int(parts[3]),
        "day": int(parts[4]),
        "hour": int(parts[5]),
        "minute": int(parts[6]),
        "second": int(parts[7])
    }

# === Process all images and compute relative luminance ===

image_dir = r"C:\Users\artur\Documents\ASI Solar Analytics\Initial_Months_Data\Processed_Images"
image_files = glob.glob(os.path.join(image_dir, "*.jpg"))
results = []

for image_path in image_files:
    img = cv2.imread(image_path)
    if img is None:
        continue
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    shape = img_rgb.shape
    name = os.path.basename(image_path)
    sun_center = detect_sun(img_rgb)

    if sun_center is None:
        sun_vec = np.array([0, 0, 1])
    else:
        sun_vec = solar_direction(sun_center[0], sun_center[1], shape)

    points = hemispherical_sampling()
    R = rotation_matrix(np.array([0, 0, 1]), sun_vec)
    points_rotated = points @ R.T
    proj = project_points(shape, points_rotated)
    rgb_values = extract_rgb(img_rgb, proj)
    luminance = compute_luminance(rgb_values)

    info = extract_image_metadata(name)
    info.update({
        "image": name,
        "relative_luminance": luminance
    })
    results.append(info)

df_luminance = pd.DataFrame(results)
df_luminance.to_csv("relative_luminance_detailed.csv", index=False)
print(df_luminance.head())

# === Calculate solar zenith angle ===

df = pd.read_csv("relative_luminance_detailed.csv")
latitude = -5.7945
longitude = -35.2110
timezone = "America/Fortaleza"

zenith_angles = []
for idx, row in df.iterrows():
    local_time = datetime(row["year"], row["month"], row["day"], row["hour"], row["minute"])
    local_time = pytz.timezone(timezone).localize(local_time)
    solar_pos = get_solarposition(local_time, latitude, longitude)
    zenith = solar_pos.iloc[0]['zenith']
    zenith_angles.append(zenith)

df["zenith_angle"] = zenith_angles
df.to_csv("luminance_with_zenith.csv", index=False)
print(df[["image", "date", "time", "relative_luminance", "zenith_angle"]].head())

# === Cosine correction ===

df = pd.read_csv("luminance_with_zenith.csv")
df["cos_zenith"] = np.cos(np.radians(df["zenith_angle"]))
df["L"] = df["relative_luminance"] * df["cos_zenith"]
df.to_csv("corrected_luminance_data.csv", index=False)
print(df[["image", "relative_luminance", "zenith_angle", "cos_zenith", "L"]].head())

# === Final merge with ground-truth solar station data ===

station_data_path = r"C:\Users\artur\Documents\ASI Solar Analytics\spreadsheets\Processed_SolarStation_Data_2015.csv"
image_data_path = "corrected_luminance_data.csv"

df_station = pd.read_csv(station_data_path, sep=';', encoding='utf-8')
df = pd.read_csv(image_data_path)

# Harmonize time columns
if df['time'].dtype == object:
    df = df.rename(columns={'time': 'time_str'})
if 'hour' in df.columns:
    df = df.rename(columns={"hour": "hour"})

if df_station['hour'].dtype == object:
    df_station[['hour', 'minute']] = df_station['hour'].str.split(':', expand=True)
    df_station['hour'] = df_station['hour'].astype(int)
    df_station['minute'] = df_station['minute'].astype(int)

keys = ["year", "month", "day", "hour", "minute"]
df[keys] = df[keys].astype(int)
df_station[keys] = df_station[keys].astype(int)

df_merged = pd.merge(df_station, df, on=keys, how="inner")
final_output = r"C:\Users\artur\Documents\ASI Solar Analytics\spreadsheets\Merged_SolarData_2015.csv"
df_merged.to_csv(final_output, index=False, sep=';', encoding='utf-8', float_format='%.10f')

print("✅ Merge completed successfully!")
print(f"📁 Saved to: {final_output}")
print(df_merged.head())
