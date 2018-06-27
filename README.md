# Energy-Forecasting-FULL-PIPELINE
Data Scraper, FBProphet, XGBoost

Full data pipeline that includes pulling historical energy/weather data (XML parsing, wget csvs), feature engineering (pandas/numpy), building forecast (FBProphet) and classification (XGBoost) models and visualizing output.

main.py:
-This file pulls historical data and creates a 24 hour energy, temp, humitiy, press forecast for tomorrow, then uses these values in a classifcation model to say if its a top 5 demand hour or not.

![alt text](https://i.gyazo.com/bb5d9112c1f6e37109f13d190ec2e1cc.png)

main_v2.py:
-This files pulls IESO forecasts and openweathermap API forecasts and uses these in the classifcaiton model to predict demand days for the next 6 days.

data_scraper:
-pulls ieso and statscan HISTORICAL data

forecast_scraper:
-pulls IESO forecast data (XML parsing)

openweatherapp_forecast:
-uses openweatherapp API to pull 5 day weather forecast (3 hour resoluition, interpolated for full 24hrs a day)

