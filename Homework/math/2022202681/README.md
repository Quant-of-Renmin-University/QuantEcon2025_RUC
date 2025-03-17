# Housing Price-to-Rent Ratios Analyzing 

## Project Overview

This project aims to collect, process, and analyze housing price and rental data for various blocks in Beijing's Haidian district. The key objectives include:

1. **Data Collection**: Gather housing price and rental information from multiple sources.
2. **Data Processing**: Merge the collected data and perform descriptive analysis.
3. **Advanced Analysis**: Calculate median price-to-rent ratios and use regression models to predict prices and rents for specified square meters.

## Table of Contents

- [Housing Price-to-Rent Ratios Analyzing](#housing-price-to-rent-ratios-analyzing)
  - [Project Overview](#project-overview)
  - [Table of Contents](#table-of-contents)
  - [Data Collection](#data-collection)
    - [1.1 Collect Rental Prices](#11-collect-rental-prices)
    - [1.2 Collect Housing Prices](#12-collect-housing-prices)
  - [Data Processing](#data-processing)
    - [2.1 Data Merging](#21-data-merging)
    - [2.2 Data Description](#22-data-description)
  - [Advanced Analysis](#advanced-analysis)
    - [3.1 Median Price-to-Rent Ratio (Fig A.)](#31-median-price-to-rent-ratio-fig-a)
    - [3.2 Predicted Price-to-Rent Ratios for m² = 50, 100 (Fig B. and C.)](#32-predicted-price-to-rent-ratios-for-m--50-100-fig-b-and-c)
  - [Dependencies](#dependencies)
  - [Usage](#usage)
  - [Conclusion](#conclusion)

---

## Data Collection

### 1.1 Collect Rental Prices

The rental data was scraped from a real estate website using Selenium WebDriver. The script navigates through pages of listings in the Haidian district, specifically targeting areas like Suzhou Bridge. It extracts area sizes (in square meters) and monthly rents.

- **Tool Used**: Selenium WebDriver with Edge browser.
- **Data Saved**: `rental_info_苏州桥.csv`

### 1.2 Collect Housing Prices

Similarly, housing prices were collected from another section of the same real estate platform. The script captured area sizes and price per square meter for resale properties in the same regions.

- **Tool Used**: Selenium WebDriver with Edge browser.
- **Data Saved**: `resold_info_苏州桥.csv`

---

## Data Processing

### 2.1 Data Merging

After collecting raw data, it was processed and merged into unified datasets for further analysis:

- **Rental Data**: Combined datasets from Suzhou Bridge, Shijicheng, Beitaipingzhuang, and Wanliu.
  - **Saved File**: `merged_rental_data.csv`
  
- **Resale Data**: Similarly combined datasets for the same regions.
  - **Saved File**: `merged_resold_data.csv`

### 2.2 Data Description

Descriptive statistics were generated to understand data distributions, detect outliers, and identify missing values. Key findings include:

- **Outlier Detection**: Utilized IQR method to detect and visualize outliers using boxplots.
- **Missing Values**: Checked for null entries across datasets.
- **Summary Statistics**: Basic statistical descriptions including mean, median, and standard deviation.

---

## Advanced Analysis

### 3.1 Median Price-to-Rent Ratio (Fig A.)

Calculated the median price-to-rent ratio for each block:

```python
price_per_m2 = resold_data.groupby('location')['unit_price'].median()
rent_per_m2 = rental_data.groupby('location')['unit_rent'].median()
price_to_rent_ratio = price_per_m2 / rent_per_m2
```

- **Visualization**: Bar plot showing median price-to-rent ratios by location.

### 3.2 Predicted Price-to-Rent Ratios for m² = 50, 100 (Fig B. and C.)

Two regression models were used to predict housing prices and rents:

**Model 1:**
\[ \text{price/m}^2_i = \beta_0\text{m}^2_i + \beta_1\text{location}_i + \beta_2\text{m}^2_i \times \text{location}_i + \epsilon_i \]

**Model 2:**
\[ \text{rent/m}^2_i = \beta_0\text{m}^2_i + \beta_1\text{location}_i + \beta_2\text{m}^2_i \times \text{location}_i + \epsilon_i \]

- **Predictions**: Generated price-to-rent ratios for \( m^2 = 50 \) and \( m^2 = 100 \).
- **Visualization**: Bar plots displaying predicted ratios grouped by location and square meter values.

---

## Dependencies

- Python 3.x
- Libraries:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - statsmodels
  - selenium

## Usage

1. Clone the repository.
2. Install required dependencies using `pip install -r requirements.txt`.
3. Run the scripts in sequence as described in the sections above.

## Conclusion

This project provides an end-to-end pipeline for collecting, processing, and analyzing housing market data. The results offer insights into regional price-to-rent dynamics and predictive modeling capabilities for specified property sizes.