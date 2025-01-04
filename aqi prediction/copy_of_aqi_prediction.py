import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import matplotlib
import datetime
import missingno as msno
from sklearn.impute import KNNImputer

from pandas.plotting import lag_plot, autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import adfuller, acf, pacf,arma_order_select_ic

df= pd.read_csv(r'C:\Users\Chaitanya\PycharmProjects\AIproj\city_day.csv',parse_dates=True).sort_values(by = ['Date'])

df.tail(5)

df.isnull().sum()

df['PM2.5']=df['PM2.5'].fillna((df['PM2.5'].median()))
df['PM10']=df['PM10'].fillna((df['PM10'].median()))
df['NO']=df['NO'].fillna((df['NO'].median()))
df['NO2']=df['NO2'].fillna((df['NO2'].median()))
df['NOx']=df['NOx'].fillna((df['NOx'].median()))
df['NH3']=df['NH3'].fillna((df['NH3'].median()))
df['CO']=df['CO'].fillna((df['CO'].median()))
df['SO2']=df['SO2'].fillna((df['SO2'].median()))
df['O3']=df['O3'].fillna((df['O3'].median()))
df['Benzene']=df['Benzene'].fillna((df['Benzene'].median()))
df['Toluene']=df['Toluene'].fillna((df['Toluene'].median()))
df['Xylene']=df['Xylene'].fillna((df['Xylene'].median()))
df['AQI']=df['AQI'].fillna((df['AQI'].median()))

df.isnull().sum()


import streamlit as st
from PIL import Image

st.title("AQI Predictor")
st.sidebar.title("Select Options")
background = '''
<style>
body {
    background-image: url("https://example.com/your-background-image.jpg");
    background-size: cover;
}
</style>
'''

# Inject the custom CSS
st.markdown(background, unsafe_allow_html=True)
city_selected = st.sidebar.selectbox("Select City", ['Chennai', 'Delhi','Mumbai', 'Bengaluru','Hyderabad'])

forcast_period = st.sidebar.selectbox("Forcast Period Days", [182,365,547,730])

month_selected = st.sidebar.selectbox("Select Month", ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'])

date_selected = st.sidebar.selectbox("Select Date", ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '23', '24', '25', '26', '27', '28', '29', '30', '31']
)

# You can add more widgets/options in the sidebar as per your requirement


# Process button
if st.sidebar.button("GO"):

    st.write("Selected City:", city_selected)
    st.write("Selected Date:", f"{month_selected}-{date_selected}")
    st.write("Forcast ",forcast_period," days into the future")
    citydf = df.loc[df['City'] == city_selected]
    citydf['Date'] = pd.to_datetime(citydf['Date'])
    citydf.set_index('Date', inplace=True)

    c = ['City', 'Benzene', 'Toluene', 'Xylene', 'AQI_Bucket', 'PM10']
    # Removing unneccesary columns
    citydf.drop(c, axis=1, inplace=True)
    citydf = citydf.sort_values('Date')
    citydf = citydf[citydf.index >= '2018-01-01']

    st.line_chart(citydf)

    from statsmodels.tsa.seasonal import seasonal_decompose

    decomposition = seasonal_decompose(citydf['AQI'], model='additive', period=365)

    # Plotting
    fig, axes = plt.subplots(4, 1, figsize=(10, 8))  # 4 rows for seasonal decomposition plot
    decomposition.observed.plot(ax=axes[0], title='Observed')
    decomposition.trend.plot(ax=axes[1], title='Trend')
    decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
    decomposition.resid.plot(ax=axes[3], title='Residual')

    # Display plot in Streamlit
    st.pyplot(fig)

    train = citydf[:'2019-12-31']
    test = citydf['2020-01-01':]

    from statsmodels.tsa.arima.model import ARIMA
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error

    # Assume 'AQI' is the target variable, and other factors are exogenous variables
    exogenous_cols = ['NO2', 'CO', 'SO2', 'PM2.5', 'O3']
    endogenous_col = 'AQI'
    # Fit ARIMA model
    # Specify the order as (p, d, q), where p is the AR order, d is the differencing order, and q is the MA order
    # Adjust these values based on the ACF and PACF plots
    p = 2  # AR order
    d = 1  # Differencing order (if needed)
    q = 2  # MA order

    # Specify the exogenous variables
    exog_train = train[exogenous_cols]
    exog_test = test[exogenous_cols]
    endog_train = train[endogenous_col]
    endog_test = test[endogenous_col]

    # Fit ARIMA model with exogenous variables
    model = ARIMA(train['AQI'], exog=exog_train, order=(p, d, q))
    results = model.fit()

    # Make predictions

    forecast_steps = len(test)
    forecast = results.get_forecast(steps=forecast_steps, exog=exog_test)
    predicted_values = forecast.predicted_mean
    print(predicted_values)

    mse = mean_squared_error(test['AQI'], predicted_values)
    print(f"Mean Squared Error(MSE): {mse}")
    rmse = np.sqrt(mse)
    print(f"Root Mean Squared Error (RMSE): {rmse}")

    # Plot results
    fig, ax = plt.subplots()
    ax.plot(test.index, endog_test, label='Observed', color='black')
    ax.plot(predicted_values.index, predicted_values, label='Forecast', color='red')
    ax.legend()
    ax.set_title('AQI Prediction with ARIMA Model and Exogenous Variables')
    # Display plot in Streamlit
    st.pyplot(fig)

    import xgboost as xgb

    FEATURES = ['PM2.5', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3']
    X_train = train[FEATURES]
    y_train = train['AQI']

    X_test = test[FEATURES]
    y_test = test['AQI']
    reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree',
                           n_estimators=1000,
                           early_stopping_rounds=50,
                           objective='reg:linear',
                           max_depth=3,
                           learning_rate=0.01)
    reg.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=100)

    test['prediction'] = reg.predict(X_test)
    plt.plot(test.index, test['AQI'], label='Observed', color='black')
    plt.plot(test.index, test['prediction'], label='Forecast', color='red')
    plt.legend()
    plt.figure(figsize=(25, 12))
    plt.show()

    score = (mean_squared_error(test['AQI'], test['prediction']))
    rmse_sc = np.sqrt(score)
    print(f'MSE Score on Test set: {score:0.2f}')
    print(f'RMSE Score on Test set: {rmse_sc:0.2f}')
#   """**Predicting for the next 1 year, 3 years and 5 years respectively**"""

    prd = forcast_period #int(input("Enter time period for forecast- 182 days, 365 days, or 547 days : "))

    # using prophet to predict pollutant concentrations
    from prophet import Prophet

    m = Prophet()
    df2 = pd.DataFrame()
    df2['ds'] = train.index
    ls = list(train['PM2.5'])
    df2.insert(1, "y", ls, allow_duplicates=True)
    m.fit(df2)
    future1 = m.make_future_dataframe(periods=prd)

    forecast1 = m.predict(future1)
    test_PM_25 = pd.DataFrame(forecast1[730::])

    m = Prophet()
    df2 = pd.DataFrame()
    df2['ds'] = train.index
    ls = list(train['O3'])
    df2.insert(1, "y", ls, allow_duplicates=True)
    m.fit(df2)
    future1 = m.make_future_dataframe(periods=prd)
    forecast1 = m.predict(future1)
    test_O3 = pd.DataFrame(forecast1[730::])

    m = Prophet()
    df2 = pd.DataFrame()
    df2['ds'] = train.index
    ls = list(train['CO'])
    df2.insert(1, "y", ls, allow_duplicates=True)
    m.fit(df2)
    future1 = m.make_future_dataframe(periods=prd)

    forecast1 = m.predict(future1)
    test_CO = pd.DataFrame(forecast1[730::])

    m = Prophet()
    df2 = pd.DataFrame()
    df2['ds'] = train.index
    ls = list(train['SO2'])
    df2.insert(1, "y", ls, allow_duplicates=True)
    m.fit(df2)
    future1 = m.make_future_dataframe(periods=prd)

    forecast1 = m.predict(future1)
    test_SO2 = pd.DataFrame(forecast1[730::])

    m = Prophet()
    df2 = pd.DataFrame()
    df2['ds'] = train.index
    ls = list(train['NO2'])
    df2.insert(1, "y", ls, allow_duplicates=True)
    m.fit(df2)
    future1 = m.make_future_dataframe(periods=prd)
    forecast1 = m.predict(future1)
    test_NO2 = pd.DataFrame(forecast1[730::])

    yhat_columns = {
        'NO2': test_NO2['yhat'],
        'PM_25': test_PM_25['yhat'],
        'SO2': test_SO2['yhat'],
        'CO': test_CO['yhat'],
        'O3': test_O3['yhat']
    }

    # creating future dataframe
    future = pd.concat(yhat_columns, axis=1)
    future.index = test_NO2['ds']
    future.index.rename('Date', inplace=True)
    future.columns = ['NO2', 'PM2.5', 'SO2', 'CO', 'O3']
    # Using xgboost to predict values
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error

    # Prepare training data
    X_train = train[['NO2', 'PM2.5', 'SO2', 'CO', 'O3']]  # Use historical pollutant concentrations as features
    y_train = train['AQI']
    print(train)
    model = xgb.XGBRegressor()
    model.fit(X_train, y_train)

    # Make predictions for the next one year using predicted pollutant concentrations (future)
    future_predictions = model.predict(future)
    plt.figure(figsize=(10, 6))
    plt.plot(future.index, future_predictions, label='Predicted AQI', color='red')
    plt.xlabel('Date')
    plt.ylabel('AQI')
    plt.title('Predicted AQI for ')
    plt.legend()
    # Display plot in Streamlit
    st.pyplot(plt.gcf())
    specific_date = '2020-01-01'  # Change this to your desired date
    specific_date = pd.to_datetime(specific_date)
    # Check if the specific date exists in the index
    prediction = pd.DataFrame(zip(future1['ds'][len(train)-1::], future_predictions), columns=['Date', 'Prediction'])
    prediction.set_index('Date', inplace=True)
    print(future1.dtypes)
    #    print(prediction)
    if specific_date in prediction.index:
        # Get the prediction for the specific date
        prediction_for_specific_date = prediction.loc[specific_date, 'Prediction']
        # Print the prediction for the specific date
        st.write(f"Prediction for {specific_date}: {prediction_for_specific_date}")
    else:
        st.write(f"No prediction available for {specific_date}.")
