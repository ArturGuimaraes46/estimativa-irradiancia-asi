import cv2
import numpy as np
import os
import pandas as pd

# === Paths ===
input_folder = r"C:\Users\artur\Documents\ASI Solar Analytics\Initial_Months_Data\Processed_Images"
sun_output_folder = r"C:\Users\artur\Documents\ASI Solar Analytics\Initial_Months_Data\Sun_Detected"
sky_output_folder = r"C:\Users\artur\Documents\ASI Solar Analytics\Initial_Months_Data\Sky_Detected_BGR"
classified_output_folder = r"C:\Users\artur\Documents\ASI Solar Analytics\Initial_Months_Data\Full_Classification"
csv_output_folder = r"C:\Users\artur\Documents\ASI Solar Analytics\spreadsheets"
os.makedirs(sun_output_folder, exist_ok=True)
os.makedirs(sky_output_folder, exist_ok=True)
os.makedirs(classified_output_folder, exist_ok=True)
os.makedirs(csv_output_folder, exist_ok=True)

# === Sun detection and highlight ===
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[ERROR] Failed to load: {filename}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        _, _, v_channel = cv2.split(hsv)

        # Brightness threshold (tuneable)
        _, thresh = cv2.threshold(v_channel, 240, 255, cv2.THRESH_BINARY)

        # Morphological cleaning
        kernel = np.ones((5, 5), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.dilate(cleaned, kernel, iterations=1)

        # Detect contours and extract the largest (sun)
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sun_mask = np.zeros_like(v_channel)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            cv2.drawContours(sun_mask, [largest], -1, 255, -1)

        # Highlight sun in yellow
        sun_highlighted = img_rgb.copy()
        sun_highlighted[sun_mask == 255] = [255, 255, 0]

        # Save outputs
        base_name = os.path.splitext(filename)[0]
        cv2.imwrite(os.path.join(sun_output_folder, f"{base_name}_sun_mask.png"), sun_mask)
        cv2.imwrite(os.path.join(sun_output_folder, f"{base_name}_sun_highlighted.png"), cv2.cvtColor(sun_highlighted, cv2.COLOR_RGB2BGR))
        print(f"[OK] Sun detected in: {filename}")

# === Sky detection based on B - G difference ===
sky_threshold = 20

for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[ERROR] Failed to load: {filename}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W, _ = img_rgb.shape
        R = img_rgb[:, :, 0].astype(float)
        G = img_rgb[:, :, 1].astype(float)
        B = img_rgb[:, :, 2].astype(float)

        # B - G difference
        b_minus_g = B - G

        # Valid pixels mask (exclude black)
        valid_mask = (R != 0) | (G != 0) | (B != 0)

        # Sky detection
        sky_mask = np.zeros((H, W), dtype=np.uint8)
        sky_mask[(b_minus_g > sky_threshold) & valid_mask] = 255

        # Visualize sky in light blue
        sky_highlighted = img_rgb.copy()
        sky_highlighted[sky_mask == 255] = [135, 206, 235]

        base = os.path.splitext(filename)[0]
        cv2.imwrite(os.path.join(sky_output_folder, f"{base}_sky_mask_bgr.png"), sky_mask)
        cv2.imwrite(os.path.join(sky_output_folder, f"{base}_sky_highlighted_bgr.png"), cv2.cvtColor(sky_highlighted, cv2.COLOR_RGB2BGR))
        print(f"[OK] Sky detected in: {filename}")

# === Full pixel classification: sun, sky, cloud ===
sun_thresh = 240
sky_thresh = 20
classification_stats = []

for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[ERROR] Failed to load: {filename}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W, _ = img_rgb.shape
        total_pixels = H * W
        R = img_rgb[:, :, 0].astype(float)
        G = img_rgb[:, :, 1].astype(float)
        B = img_rgb[:, :, 2].astype(float)
        valid_mask = (R != 0) | (G != 0) | (B != 0)

        # Sun detection using V from HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        _, _, V = cv2.split(hsv)
        sun_mask = (V > sun_thresh) & valid_mask

        # Sky detection: (B - G) > threshold and not sun
        sky_mask = (B - G > sky_thresh) & valid_mask & (~sun_mask)

        # Cloud detection: everything else valid
        cloud_mask = valid_mask & (~sun_mask) & (~sky_mask)

        # Pixel counts
        valid_pixels = np.sum(valid_mask)
        n_sun = np.sum(sun_mask)
        n_sky = np.sum(sky_mask)
        n_cloud = np.sum(cloud_mask)

        perc_sun = (n_sun / valid_pixels) * 100
        perc_sky = (n_sky / valid_pixels) * 100
        perc_cloud = (n_cloud / valid_pixels) * 100

        classification_stats.append({
            "image": filename,
            "valid_pixels": valid_pixels,
            "sun_%": round(perc_sun, 2),
            "sky_%": round(perc_sky, 2),
            "cloud_%": round(perc_cloud, 2)
        })

        # Save classification visualization
        classified = img_rgb.copy()
        classified[sun_mask] = [255, 255, 0]       # Yellow = Sun
        classified[sky_mask] = [135, 206, 235]     # Light blue = Sky
        classified[cloud_mask] = [220, 220, 220]   # Light gray = Clouds

        output_path = os.path.join(classified_output_folder, f"{os.path.splitext(filename)[0]}_classified.png")
        cv2.imwrite(output_path, cv2.cvtColor(classified, cv2.COLOR_RGB2BGR))

        print(f"[OK] {filename} → Sun: {perc_sun:.2f}%, Sky: {perc_sky:.2f}%, Cloud: {perc_cloud:.2f}%")

# === Save statistics to CSV ===
df_stats = pd.DataFrame(classification_stats)
csv_path = os.path.join(csv_output_folder, "pixel_classification_percentages.csv")
df_stats.to_csv(csv_path, index=False)
print(f"\n✅ Classification statistics saved to: {csv_path}")
