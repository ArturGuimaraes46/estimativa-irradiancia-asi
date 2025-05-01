import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr

# === 1. Plot polynomial models by cloud cover range ===
def plot_polynomial_models(df, luminance_col, irradiance_col, cloud_col):
    """
    Fit and plot 4th-degree polynomial models for each 10% cloud coverage range,
    showing data points and fitted curves with R², RMSE and Spearman correlation.
    """
    intervals = [(i, i+10) for i in range(0, 100, 10)]
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    plt.figure(figsize=(14, 9))

    for idx, (cmin, cmax) in enumerate(intervals):
        subset = df[(df[cloud_col] >= cmin) & (df[cloud_col] < cmax)]
        if subset.empty:
            continue

        X = subset[[luminance_col]].values
        y = subset[irradiance_col].values

        poly = PolynomialFeatures(degree=4)
        X_poly = poly.fit_transform(X)
        model = LinearRegression().fit(X_poly, y)
        y_pred = model.predict(X_poly)

        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        spearman_corr, _ = spearmanr(y, y_pred)

        x_seq = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
        y_seq = model.predict(poly.transform(x_seq))

        plt.scatter(X, y, color=colors[idx], edgecolor='k', alpha=0.6,
                    label=f'{cmin}-{cmax}% cloud\nR²={r2:.2f} | RMSE={rmse:.1f} | ρ={spearman_corr:.2f}')
        plt.plot(x_seq, y_seq, color=colors[idx], linewidth=2)

    plt.title("4th-Degree Polynomial Models by Cloud Coverage Range", fontsize=16)
    plt.xlabel("Corrected Luminance (L)", fontsize=14)
    plt.ylabel("Global Irradiance (W/m²)", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title="Cloud Coverage & Metrics", fontsize=10, loc='upper left', bbox_to_anchor=(1.02, 1))
    plt.tight_layout()
    plt.show()

# === 2. Plot real vs predicted irradiance by cloud band ===
def plot_real_vs_predicted(df, luminance_col, irradiance_col, cloud_col):
    intervals = [(i, i+10) for i in range(0, 100, 10)]
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    plt.figure(figsize=(15, 10))

    for idx, (cmin, cmax) in enumerate(intervals):
        subset = df[(df[cloud_col] >= cmin) & (df[cloud_col] < cmax)]
        if subset.empty:
            continue

        X = subset[[luminance_col]].values
        y = subset[irradiance_col].values
        poly = PolynomialFeatures(degree=4)
        X_poly = poly.fit_transform(X)
        model = LinearRegression().fit(X_poly, y)
        y_pred = model.predict(X_poly)

        order = np.argsort(X.flatten())
        plt.subplot(5, 2, idx+1)
        plt.plot(X.flatten()[order], y[order], '--', label='Real', color='black')
        plt.plot(X.flatten()[order], y_pred[order], label='Predicted', color=colors[idx])
        plt.title(f'{cmin}-{cmax}% Cloud')
        plt.xlabel('Corrected Luminance (L)')
        plt.ylabel('Global Irradiance (W/m²)')
        plt.grid(True)

        if idx == 0:
            plt.legend()

    plt.tight_layout()
    plt.suptitle('Real vs Predicted by Cloud Range (4th-Degree Polynomial)', fontsize=18, y=1.02)
    plt.show()

# === 3. Compute metrics by cloud interval ===
def compute_metrics_by_cloud_range(df, luminance_col, irradiance_col, cloud_col):
    intervals = [(i, i+10) for i in range(0, 100, 10)]
    results = []

    for cmin, cmax in intervals:
        subset = df[(df[cloud_col] >= cmin) & (df[cloud_col] < cmax)]
        if subset.empty:
            continue

        X = subset[[luminance_col]].values
        y = subset[irradiance_col].values
        poly = PolynomialFeatures(degree=4)
        model = LinearRegression().fit(poly.fit_transform(X), y)
        y_pred = model.predict(poly.transform(X))

        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        spearman_corr, _ = spearmanr(y, y_pred)

        results.append({
            "Cloud Range (%)": f"{cmin}-{cmax}",
            "R²": round(r2, 4),
            "MAE (W/m²)": round(mae, 2),
            "RMSE (W/m²)": round(rmse, 2),
            "Spearman ρ": round(spearman_corr, 4)
        })

    return pd.DataFrame(results)

# === 4. Predict per point and add modeled irradiance column ===
def apply_polynomial_model(df, luminance_col, irradiance_col, cloud_col):
    df['Global_Avg_Predicted'] = np.nan
    intervals = [(i, i+10) for i in range(0, 100, 10)]

    for cmin, cmax in intervals:
        subset = df[(df[cloud_col] >= cmin) & (df[cloud_col] < cmax)]
        if subset.empty:
            continue

        X = subset[[luminance_col]].values
        y = subset[irradiance_col].values
        poly = PolynomialFeatures(degree=4)
        model = LinearRegression().fit(poly.fit_transform(X), y)
        y_pred = model.predict(poly.transform(X))
        df.loc[subset.index, 'Global_Avg_Predicted'] = y_pred

    return df

# === 5. Global performance statistics ===
def compute_overall_statistics(df, real_col, predicted_col):
    df_clean = df.dropna(subset=[real_col, predicted_col])
    y_true = df_clean[real_col].values
    y_pred = df_clean[predicted_col].values

    return {
        "R²": round(r2_score(y_true, y_pred), 4),
        "MAE (W/m²)": round(mean_absolute_error(y_true, y_pred), 2),
        "RMSE (W/m²)": round(np.sqrt(mean_squared_error(y_true, y_pred)), 2),
        "Spearman ρ": round(spearmanr(y_true, y_pred)[0], 4)
    }

# === 6. Diagnostic plots ===
def plot_model_diagnostics(df, real_col, predicted_col):
    df_clean = df.dropna(subset=[real_col, predicted_col])
    residuals = df_clean[real_col] - df_clean[predicted_col]

    # Scatter
    plt.figure(figsize=(8, 6))
    plt.scatter(df_clean[real_col], df_clean[predicted_col], alpha=0.5, edgecolors='k')
    plt.plot([df_clean[real_col].min(), df_clean[real_col].max()],
             [df_clean[real_col].min(), df_clean[real_col].max()],
             'r--', label='Ideal')
    plt.title("Scatter: Measured vs Predicted")
    plt.xlabel("Measured Global Irradiance (W/m²)")
    plt.ylabel("Predicted Global Irradiance (W/m²)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Residuals
    plt.figure(figsize=(8, 6))
    plt.scatter(df_clean[real_col], residuals, alpha=0.5, edgecolors='k')
    plt.axhline(0, color='red', linestyle='--')
    plt.title("Residuals: Measured - Predicted")
    plt.xlabel("Measured Irradiance (W/m²)")
    plt.ylabel("Residual (W/m²)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Residual distribution
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True, color='skyblue', edgecolor='black')
    plt.axvline(0, color='red', linestyle='--')
    plt.title("Residual Distribution")
    plt.xlabel("Residual (W/m²)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# === Execution ===
csv_path = r"C:\Users\artur\Documents\ASI Solar Analytics\spreadsheets\Merged_Dataset_Final.csv"
df = pd.read_csv(csv_path, sep=';')

# Apply modeling
df_modeled = apply_polynomial_model(df, 'L_Normalized', 'Global_Avg', 'cloud_%')

# Save modeled output
df_modeled.to_csv(r"C:\Users\artur\Documents\ASI Solar Analytics\spreadsheets\Modeled_Dataset.csv", sep=';', index=False)

# Print overall metrics
metrics = compute_overall_statistics(df_modeled, 'Global_Avg', 'Global_Avg_Predicted')
print("\n📊 Overall Modeling Metrics:\n", metrics)

# Visual diagnostics
plot_model_diagnostics(df_modeled, 'Global_Avg', 'Global_Avg_Predicted')
