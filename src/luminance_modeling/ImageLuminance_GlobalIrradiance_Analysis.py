import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import spearmanr

# === Function to merge classification percentages and solar station data ===
def merge_datasets(path_classification, path_station, key='image'):
    """
    Merge two datasets from CSV files using a common key.

    Parameters:
    - path_classification (str): CSV file with cloud/sky/sun pixel percentages.
    - path_station (str): CSV file with solar irradiance data.
    - key (str): Key column used for merging.

    Returns:
    - merged_df (DataFrame): Resulting merged DataFrame.
    """
    df_class = pd.read_csv(path_classification)
    df_station = pd.read_csv(path_station, sep=';', encoding='utf-8')
    merged_df = pd.merge(df_station, df_class, how='left', on=key)
    return merged_df

# === File paths ===
classification_csv = r"C:\Users\artur\Documents\ASI Solar Analytics\spreadsheets\pixel_classification_percentages.csv"
station_csv = r"C:\Users\artur\Documents\ASI Solar Analytics\spreadsheets\Merged_SolarData_2015.csv"

# === Merge data ===
merged_df = merge_datasets(classification_csv, station_csv)

# === Normalization factor (Dev et al.) ===
def compute_normalization_factor(df, irradiance_col, luminance_col):
    a = df[irradiance_col].values
    b = df[luminance_col].values
    return np.sum(a * b) / np.sum(b ** 2)

norm_factor = compute_normalization_factor(merged_df, 'Global_Avg', 'L')
print(f"🔁 Normalization factor: {norm_factor:.4f}")

# === Apply normalization ===
merged_df['L_Normalized'] = merged_df['L'] * norm_factor

# === Compute evaluation metrics ===
def compute_statistics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    spearman_corr, _ = spearmanr(y_true, y_pred)
    return {
        "R²": round(r2, 4),
        "MAE (W/m²)": round(mae, 2),
        "RMSE (W/m²)": round(rmse, 2),
        "Spearman Correlation (ρ)": round(spearman_corr, 4)
    }

metrics = compute_statistics(merged_df['Global_Avg'], merged_df['L_Normalized'])
print("\n📊 Evaluation Metrics (Global_Avg vs L_Normalized):\n", metrics)

# === Scatter plot: Measured vs Modeled ===
plt.figure(figsize=(8, 6))
plt.scatter(merged_df['Global_Avg'], merged_df['L_Normalized'], alpha=0.5, edgecolors='k')
plt.plot([merged_df['Global_Avg'].min(), merged_df['Global_Avg'].max()],
         [merged_df['Global_Avg'].min(), merged_df['Global_Avg'].max()],
         color='red', linestyle='--', label='Ideal (y = x)')
plt.title('Scatter Plot: Global_Avg vs L_Normalized (Dev et al. factor)')
plt.xlabel('Measured Global Irradiance (W/m²)')
plt.ylabel('Normalized Luminance (W/m²)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Optional: Save merged database ===
merged_df.to_csv(r"C:\Users\artur\Documents\ASI Solar Analytics\spreadsheets\Merged_Dataset_Final.csv",
                 index=False, sep=';', float_format='%.10f')
print("✅ Final dataset saved successfully!")

# === Histogram Plot ===
def plot_histograms(df, columns, bins=30):
    for col in columns:
        plt.figure(figsize=(8, 5))
        plt.hist(df[col].dropna(), bins=bins, edgecolor='black', alpha=0.7)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.show()

plot_histograms(merged_df, ['sun_%', 'cloud_%', 'sky_%', 'Global_Avg'])

# === Scatter Plot between two variables ===
def plot_scatter(df, x_col, y_col):
    plt.figure(figsize=(8, 6))
    plt.scatter(df[x_col], df[y_col], alpha=0.5)
    plt.title(f"Scatter Plot: {x_col} vs {y_col}")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.grid(True)
    plt.show()

plot_scatter(merged_df, 'L', 'Global_Avg')

# === Boxplot ===
def plot_boxplots(df, columns):
    for col in columns:
        plt.figure(figsize=(6, 4))
        plt.boxplot(df[col].dropna(), vert=True, patch_artist=True)
        plt.title(f"Boxplot of {col}")
        plt.ylabel(col)
        plt.grid(True)
        plt.show()

plot_boxplots(merged_df, ['sun_%', 'cloud_%', 'sky_%', 'Global_Avg'])

# === Correlation Matrix ===
def plot_correlation_matrix(df, columns):
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[columns].corr(), annot=True, fmt=".2f", cmap='coolwarm')
    plt.title("Correlation Matrix")
    plt.show()

plot_correlation_matrix(merged_df, ['relative_luminance', 'sun_%', 'cloud_%', 'sky_%', 'Global_Avg'])
