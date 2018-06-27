# Energy-Forecasting-FULL-PIPELINE
Data Scraper, FBProphet, XGBoost

Full pipeline that includes pulling historical energy/weather data (XML parsing, wget csvs), feature engineering (pandas/numpy), building forecast (FBProphet) and classification (XGBoost) models, forecasting/predicting and visualizing output (Seaborn, Matplotlib).

main.py (10 minute run time due to FBprophet models):
-This file pulls historical data and creates a 24 hour energy, temp, humidity, pressure forecast for tomorrow, then uses these values in a classifcation model to say if these 24 hours are a top 5 demand hour or not. Output is as follows:

![alt text](https://i.gyazo.com/f7bc70c645a2a6f3f367976b22b967ee.png)

main_v2.py[PREFERED]:
-This files pulls IESO forecasts and openweathermap API forecasts and uses these in the classifcaiton model to predict demand days for the next 6 days. Output is as follows:

![alt text](https://i.gyazo.com/bb5d9112c1f6e37109f13d190ec2e1cc.png)


Below scrapers.py are used in the main files:

data_scraper:
-pulls ieso and statscan HISTORICAL data

forecast_scraper:
-pulls IESO forecast data (XML parsing)

openweatherapp_forecast:
-uses openweatherapp API to pull 5 day weather forecast (3 hour resoluition, interpolated for full 24hrs a day)

