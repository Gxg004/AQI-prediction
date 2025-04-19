# AQI-prediction
AQI Prediction
This project involves predicting the Air Quality Index (AQI) using time series forecasting techniques, primarily focusing on the ARIMA model. The analysis includes data preprocessing, exploratory data analysis (EDA), handling of missing data, and implementation of statistical models for forecasting AQI values over time.

Dataset
The dataset used in this notebook is city_day.csv, which includes AQI and pollutant levels recorded daily across different cities in India.

Project Structure
1. Data Loading & Preprocessing
Data is loaded from Google Drive.

Datetime parsing and sorting are done to prepare the dataset for time series analysis.

Missing values are handled using KNNImputer.

2. Exploratory Data Analysis (EDA)
Visualizations using matplotlib and seaborn.

Use of missingno for visualizing missing data.

Statistical plots such as lag plots, ACF, and PACF to assess time series characteristics.

3. Stationarity Check
The Augmented Dickey-Fuller (ADF) test is used to verify stationarity, a key assumption for ARIMA modeling.

4. ARIMA Modeling
Autoregressive models (AR), moving average models (MA), and combined ARIMA models are used.

Model order is selected using arma_order_select_ic.

Model performance is evaluated using Mean Squared Error (MSE).

Libraries Used
pandas, numpy

matplotlib, seaborn, missingno

statsmodels for time series analysis

sklearn for imputation and evaluation

Output
Time series predictions of AQI.

Evaluation metrics like MSE to assess forecast accuracy.

Plots for diagnostics and model validation.
