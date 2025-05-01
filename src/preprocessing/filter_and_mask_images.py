import cv2
import numpy as np
import os
from glob import glob
import pandas as pd

# === Description ===
# This script processes all-sky images from June to October 2015.
# For each image: (i) crops a 770x770 square centered on the dome, 
# (ii) applies a circular mask (radius = 385 px), and 
# (iii) excludes files listed in a manually validated spreadsheet.
# Only valid images are saved into 'Cropped_Images' and 'Processed_Images' folders.

# === Configuration ===

# Root folder containing monthly subfolders with raw images
root_folder = r"C:\Users\artur\Documents\ASI Solar Analytics\Initial_Months_Data"

# Months to be processed
months = ['07', '08', '09', '10']

# Output directories
crops_path = os.path.join(root_folder, "Cropped_Images")
processed_path = os.path.join(root_folder, "Processed_Images")

os.makedirs(crops_path, exist_ok=True)
os.makedirs(processed_path, exist_ok=True)

# Circle parameters (mask center and radius in pixels)
circle_center = (385, 385)
circle_radius = 385

# Path to the CSV containing manually excluded filenames
csv_path = r"C:\Users\artur\Documents\ASI Solar Analytics\spreadsheets\excluded_files.csv"

# Read exclusion list
try:
    df_excluded = pd.read_csv(csv_path)
    excluded_files = df_excluded.iloc[:, 0].tolist()  # Assumes filenames are in the first column
    print(f"📄 {len(excluded_files)} files listed for exclusion.")
except Exception as e:
    print(f"❌ Error reading CSV exclusion list: {e}")
    excluded_files = []

# === Main processing loop ===
for month in months:
    month_folder = os.path.join(root_folder, f"2015_{month}")
    file_paths = glob(os.path.join(month_folder, '*.jpg'))

    for img_path in file_paths:
        original_filename = os.path.basename(img_path)
        processed_filename = f"{month}_{original_filename}"

        # Skip if listed in excluded_files
        if processed_filename in excluded_files:
            print(f"⏩ Skipped (excluded): {processed_filename}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ Failed to load image: {img_path}")
            continue

        # Crop to 770x770 pixels (adjust these bounds as needed)
        cropped_img = img[146:916, 227:997]

        if cropped_img.shape[:2] == (770, 770):
            # Save cropped image
            cropped_save_path = os.path.join(crops_path, processed_filename)
            cv2.imwrite(cropped_save_path, cropped_img)

            # Create circular mask
            mask = np.zeros((770, 770), dtype="uint8")
            cv2.circle(mask, circle_center, circle_radius, 255, -1)

            # Apply the circular mask (RGB outside becomes black)
            masked_img = np.zeros_like(cropped_img)
            for i in range(3):  # For each color channel
                masked_img[:, :, i] = cv2.bitwise_and(cropped_img[:, :, i], mask)

            # Save final masked image
            processed_save_path = os.path.join(processed_path, processed_filename)
            cv2.imwrite(processed_save_path, masked_img)
            print(f"✅ Processed and saved: {processed_save_path}")
        else:
            print(f"⚠️ Invalid crop dimensions for: {img_path}")
