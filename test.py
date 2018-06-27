import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

from fbprophet import Prophet 

from data_scraper import PullData
from forecast_scraper import pull_ieso_forecast
from openweathermap_forecast import pull_weather_forecast

import warnings
warnings.filterwarnings("ignore")

#pull forecasts
ontario_forecast = pull_ieso_forecast()

weather_forecast = pull_weather_forecast()

print('Pulling, cleaning and building forecast data sets...')

#Demand features

df = ontario_forecast.copy()

df = df.rename(columns={'Ontario': 'Ontario Demand'})

#move hours to columns
df = pd.pivot_table(df,index=['Date'],columns=['Hour'],values=['Ontario Demand'], aggfunc = 'first')


df.reset_index(inplace=True)

df['Date'] = pd.to_datetime(df['Date'])

#add max/mean/min features
df['Demand_Max'] = df.max(axis=1)
df['Demand_Mean'] = df.mean(axis=1)
df['Demand_Min'] = df.min(axis=1)

mapper = {1:'Demand_1',2:'Demand_2',3:'Demand_3',4:'Demand_4',5:'Demand_5',6:'Demand_6',7:'Demand_7',8:'Demand_8',9:'Demand_9',
          10:'Demand_10',11:'Demand_11',12:'Demand_12',13:'Demand_13',14:'Demand_14',15:'Demand_15',16:'Demand_16',17:'Demand_17',
          18:'Demand_18',19:'Demand_19',20:'Demand_20',21:'Demand_21',22:'Demand_22',23:'Demand_23',24:'Demand_24'}

df = df.rename(columns=mapper)

#Add Date features
df['Month'] = df['Date'].apply(lambda x: x.month)
df['Dayofweek'] = df['Date'].apply(lambda x: x.dayofweek)
df['Day'] = df['Date'].apply(lambda x: x.day)

#Weather features

df2 = weather_forecast.copy()


df2['Date'] = df2['Date'].dt.date
df2['Date'] = pd.to_datetime(df2['Date'])

df_temp = df2[['Date','Temp','Hour']]
df_hum = df2[['Date','Hum','Hour']]
df_press = df2[['Date','Press','Hour']]

#Hour to columns
df_temp = pd.pivot_table(df_temp,index=['Date'],columns=['Hour'],values=['Temp'])
df_hum = pd.pivot_table(df_hum,index=['Date'],columns=['Hour'],values=['Hum'])
df_press = pd.pivot_table(df_press,index=['Date'],columns=['Hour'],values=['Press'])

df_temp.reset_index(inplace=True)
df_hum.reset_index(inplace=True)
df_press.reset_index(inplace=True)


#add max/mean/min features
df_temp['temp_Max'] = df_temp.max(axis=1)
df_temp['temp_Mean'] = df_temp.mean(axis=1)
df_temp['temp_Min'] = df_temp.min(axis=1)

df_hum['hum_Max'] = df_hum.max(axis=1)
df_hum['hum_Mean'] = df_hum.mean(axis=1)
df_hum['hum_Min'] = df_hum.min(axis=1)

df_press['press_Max'] = df_press.max(axis=1)
df_press['press_Mean'] = df_press.mean(axis=1)
df_press['press_Min'] = df_press.min(axis=1)

mapper_temp = {0:'temp_0',1:'temp_1',2:'temp_2',3:'temp_3',4:'temp_4',5:'temp_5',6:'temp_6',7:'temp_7',8:'temp_8',9:'temp_9',
          10:'temp_10',11:'temp_11',12:'temp_12',13:'temp_13',14:'temp_14',15:'temp_15',16:'temp_16',17:'temp_17',
          18:'temp_18',19:'temp_19',20:'temp_20',21:'temp_21',22:'temp_22',23:'temp_23'}

mapper_hum = {0:'hum_0', 1:'hum_1',2:'hum_2',3:'hum_3',4:'hum_4',5:'hum_5',6:'hum_6',7:'hum_7',8:'hum_8',9:'hum_9',
          10:'hum_10',11:'hum_11',12:'hum_12',13:'hum_13',14:'hum_14',15:'hum_15',16:'hum_16',17:'hum_17',
          18:'hum_18',19:'hum_19',20:'hum_20',21:'hum_21',22:'hum_22',23:'hum_23'}

mapper_press = {0:'press_0', 1:'press_1',2:'press_2',3:'press_3',4:'press_4',5:'press_5',6:'press_6',7:'press_7',8:'press_8',9:'press_9',
          10:'press_10',11:'press_11',12:'press_12',13:'press_13',14:'press_14',15:'press_15',16:'press_16',17:'press_17',
          18:'press_18',19:'press_19',20:'press_20',21:'press_21',22:'press_22',23:'press_23'}


df_temp = df_temp.rename(columns=mapper_temp)
df_hum = df_hum.rename(columns=mapper_hum)
df_press = df_press.rename(columns=mapper_press)

#Merge Weather/Demand
#merge on right becuase only 5 day ahead forecasts on weather, ieso is 35 days
forecast = pd.merge(df, df_temp, how='right', on='Date')
forecast = pd.merge(forecast, df_hum, how='right', on='Date')
forecast = pd.merge(forecast, df_press, how='right', on='Date')

del forecast['Date']
forecast = forecast.rename(columns={'Temp': 'Temp (°C)', 'Hum': 'Rel Hum (%)', 'Press': 'Stn Press (kPa)'})

print(forecast)
print(forecast.info(verbose=True))


print('Done.')

print('Plotting Forecasts...')

fig = plt.figure(figsize=(100,20))
#fig.subplots_adjust(hspace=0.9, wspace=0.9)

grid = plt.GridSpec(2, 3, wspace=0.3, hspace=0.3)


plt.subplot(grid[0,0])
plt.plot(ontario_forecast['Ontario'][0:120], label = 'Max: {}'.format(ontario_forecast['Ontario'][0:120].max()), color = 'green')
plt.title('IESO 5 Day Demand Forecast')
plt.ylabel('MW')
plt.xlabel('Hour')
plt.legend(loc="upper left")

plt.subplot(grid[0,1])
plt.plot(weather_forecast['Temp'][0:120], label = 'Max: {}'.format(weather_forecast['Temp'][0:120].max()), color = 'orange')
plt.title('5 Day Temperature Forecast')
plt.ylabel('Temp (°C)')
plt.xlabel('Hour')
plt.legend(loc="upper left")

plt.subplot(grid[0,2])
plt.plot(weather_forecast['Hum'][0:120], label = 'Max: {}'.format(weather_forecast['Hum'][0:120].max()))
plt.title('5 Day Humidity Forecast')
plt.ylabel('Rel Hum (%)')
plt.xlabel('Hour')
plt.legend(loc="upper left")

plt.subplot(grid[1,0])
plt.plot(weather_forecast['Press'][0:120], label = 'Max: {}'.format(weather_forecast['Press'][0:120].max()), color = 'purple')
plt.title('5 Day Pressure Forecast')
plt.ylabel('Stn Press (kPa)')
plt.xlabel('Hour')
plt.legend(loc="upper left")

plt.show()



print('Done.')