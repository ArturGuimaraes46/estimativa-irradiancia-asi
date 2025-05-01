#  ASI Solar Irradiance Estimation

This repository contains the full pipeline developed by **Artur Guimarães**, a Master's student in Atmospheric and Climate Sciences and undergraduate in Electrical Engineering, for estimating **solar irradiance** using RGB **All-Sky images**. The project integrates methods from **computer vision**, **solar geometry**, and **statistical modeling** to quantify irradiance from sky luminance — contributing to image-based forecasting in renewable energy.

---

##  Project Objective

The goal is to estimate **Global Horizontal Irradiance (GHI)** based exclusively on visual information from All-Sky images, without the use of pyranometers or meteorological sensors.

The approach involves:

-  **RGB-based segmentation** of sky, cloud, and sun areas
-  **Luminance estimation** using hemispherical sampling and gamma correction
-  **Zenith angle correction** based on astronomical position (via `pvlib`)
-  **Polynomial modeling** (degree 4), stratified by cloud coverage bins (0–100%)
-  **Validation** against ground-truth data from a solarimetric station

The methodology was applied to data from **Natal/RN, Brazil**, a coastal region with high atmospheric variability.

---

##  Repository Structure

```plaintext
ASI-Solar-Irradiance/
+-- src/
¦   +-- preprocessing/           # Cropping, filtering and circular masking
¦   +-- segmentation/            # Classification of sky, clouds, and sun using RGB
¦   +-- luminance_modeling/      # Luminance estimation, zenith correction, merging
¦   +-- modeling/                # Polynomial regressions and metrics
¦   +-- plots/                   # Graphs and visual diagnostics
+-- data/
¦   +-- raw/                     # Original images and station spreadsheets
¦   +-- processed/               # Filtered/masked images
¦   +-- station_data/            # Ground-truth irradiance measurements (GHI, DHI, DNI)
¦   +-- outputs/                 # Final merged CSV files and modeling results
+-- notebooks/                   # Jupyter notebooks for validation and testing
+-- figures/                     # Graphs and figures for presentation/publication
+-- requirements.txt             # Python dependencies
+-- LICENSE                      # MIT license file
+-- README.md                    # This documentation
