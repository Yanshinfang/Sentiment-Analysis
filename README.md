# Sentiment-Analysis

# Data Preprocessing

1. **File Encoding and DataFrame Creation**: Convert the CSV file to UTF-8 encoding, read it into a DataFrame, and ensure that the column names in Chinese are not garbled.
2. **Date Format Conversion and Data Filtering**: Perform date format conversion and filter the data based on the specified range (10/1 - 12/31), selecting the desired dataset of over a thousand records.
3. **Handling Missing Values**: Write a `fill_missing` function to handle missing values. The function's purpose is to take the average of values before and after the missing value (until a value is found). If a missing value extends to 12/31 23:00, an additional constraint is added. If it exceeds the specified range, the function takes the average of the available values before and after.

# Time Series Analysis

## Mean Absolute Error (MAE) Comparison

1. **Linear Regression (pm2.5 only, 1-hour ahead)**: 
2. **XGBoost (pm2.5 only, 1-hour ahead)**: 
3. **Linear Regression (pm2.5 only, 6-hours ahead)**: 
4. **XGBoost (pm2.5 only, 6-hours ahead)**: 
5. **Linear Regression (all features, 1-hour ahead)**: 
6. **XGBoost (all features, 1-hour ahead)**: 
7. **Linear Regression (all features, 6-hours ahead)**: 
8. **XGBoost (all features, 6-hours ahead)**: 

As the prediction time extends, the Mean Absolute Error (MAE) generally increases. However, in the comparison between XGBoost and Linear Regression, the performance may be affected by parameter tuning or data content. In general, Linear Regression performs better. However, XGBoost outperforms Linear Regression when predicting six hours ahead with all features.
