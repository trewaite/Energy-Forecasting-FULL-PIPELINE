import numpy as np
import pandas as pd
import openweathermapy.core as own
from datetime import datetime as dt

def pull_weather_forecast():
	#get 3 hour data forecasts
	data = own.get_forecast_hourly("Toronto,CA", APPID = 'a5bbf571539ed962c45eafc7aaf47d0a', units='metric')

	#change from dt to datetime
	conv = {"dt": lambda ts: str(dt.utcfromtimestamp(ts))}

	#choose datapoints
	selection = data.select(['dt','main.temp','main.humidity','main.pressure'], converters=conv)


	#change from 3hour to hourly by interpolation
	datelist = pd.date_range(selection[0][0],periods=24*5, freq ='h')

	df1 = pd.DataFrame(datelist,columns=['Date'])
	df1.set_index('Date',drop=True,inplace=True)

	df2 = pd.DataFrame(selection,columns=['Date','Temp','Hum','Press'])
	df2.set_index('Date',drop=True,inplace=True)

	results = df1.join(df2,how='left')

	results = results.interpolate()

	#hpa to kpa
	results['Press'] = results['Press']/10

	results.reset_index(inplace = True)

	results['Hour'] = results['Date'].apply(lambda x: x.hour)

	return results

#print(pull_weather_forecast())
