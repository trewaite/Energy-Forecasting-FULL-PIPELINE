import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

from fbprophet import Prophet
import xgboost as xg
from xgboost.sklearn import XGBClassifier

from data_scraper import PullData

import warnings
warnings.filterwarnings("ignore")

#Pull Data From IESO(Demand) and StatsCan(Weather)

demand_data, weather_data = PullData() #delete old #download new

'''
...............................................................................
[START]CLEANING / FEATURE ENGINEERING / BUILDING TRAINING SETS
INPUT: demand_data, weather_data
...............................................................................

'''
print('Cleaning and building training data sets...')
#to datetime
demand_data['Date'] = pd.to_datetime(demand_data['Date']) + pd.to_timedelta(demand_data['Hour'], unit='h')
weather_data['Date/Time'] = pd.to_datetime(weather_data['Date/Time'])

#Merge
weather_data = weather_data.rename(index=str, columns = {"Date/Time":"Date"})
data = demand_data.merge(right=weather_data, how='left', on='Date')
data.drop('Time', axis = 1, inplace = True)

#Feature Creation

#add day of week (Sun-Sat)
data['Day of Week'] = data['Date'].apply(lambda x: x.dayofweek)

#add top five days (add 1 for whole day i.e 24 1's per day or 24*5 1's per year)
data.set_index('Date',drop=False, inplace = True)

top_days = 5

data['topdays'] = 0


for year in range(int(data['Year'].min()),int(data['Year'].max())+1):

    indices = data[data['Year'] == year].resample('D').max().nlargest(top_days,'Ontario Demand').index

    for i in range(len(indices)):
        
        y = data[data.index == indices[i]]['Year'].as_matrix()[0]
        m = data[data.index == indices[i]]['Month'].as_matrix()[0]
        d = data[data.index == indices[i]]['Day'].as_matrix()[0]
        h = data[data.index == indices[i]]['Hour'].as_matrix()[0]
        
        #create 4 hour band around peak with 1's (h+-3)
        data.loc[data[(data['Year'] == y) & (data['Month'] == m) & (data['Day'] == d) & (data['Hour'] >= h-3) & (data['Hour'] <= h+3)].index, 'topdays'] = 1

#Clean Data
data = data[data.columns[data.isnull().mean() < 0.80]]

#get target variable
y = data['topdays']
del data['topdays']

#remove unforecastable variables
del data['Year']
del data['Wind Dir Flag']
del data['Wind Spd Flag']

#create forecasting training sets
market = data[['Date','Market Demand']]
market.rename(index=str, columns={"Date": "ds", "Market Demand": "y"}, inplace = True)

ontario = data[['Date','Ontario Demand']]
ontario.rename(index=str, columns={"Date": "ds", "Ontario Demand": "y"}, inplace = True)

temp = data[['Date','Temp (°C)']]
temp.rename(index=str, columns={"Date": "ds", "Temp (°C)": "y"}, inplace = True)

dew = data[['Date','Dew Point Temp (°C)']]
dew.rename(index=str, columns={"Date": "ds", "Dew Point Temp (°C)": "y"}, inplace = True)

hum = data[['Date','Rel Hum (%)']]
hum.rename(index=str, columns={"Date": "ds", "Rel Hum (%)": "y"}, inplace = True)

kpa = data[['Date','Stn Press (kPa)']]
kpa.rename(index=str, columns={"Date": "ds", "Stn Press (kPa)": "y"}, inplace = True)

del data['Date']

print('Done.')

'''
...............................................................................
[END] CLEANING AND FEATURE ENGINEERING
OUTPUT: 
classification training sets: data,y
forecasting training sets: market,ontario,temp,dew,hum,kpa
...............................................................................

'''

'''
...............................................................................
[START] FBPROPHET TRAINING / FORECASTING TOMORROWS VALUES
INPUT: market,ontario,temp,dew,hum,kpa
...............................................................................

'''

#fit models
print('Fitting Forecast Models...')
market_m = Prophet()
market_m.fit(market)

ontario_m = Prophet()
ontario_m .fit(ontario)

temp_m = Prophet()
temp_m.fit(temp)

dew_m = Prophet()
dew_m.fit(dew)

hum_m = Prophet()
hum_m.fit(hum)

kpa_m = Prophet()
kpa_m.fit(kpa)
print('Done.')

#make day ahead forecats
print('Making Forecasts...')
market_future = market_m.make_future_dataframe(periods = 24, freq='H')
market_forecast = market_m.predict(market_future)
market_forecast = market_forecast.tail(24)

ontario_future = ontario_m.make_future_dataframe(periods = 24, freq='H')
ontario_forecast = ontario_m.predict(market_future)
ontario_forecast = ontario_forecast.tail(24)

temp_future = temp_m.make_future_dataframe(periods = 24, freq='H')
temp_forecast = temp_m.predict(market_future)
temp_forecast = temp_forecast.tail(24)

dew_future = dew_m.make_future_dataframe(periods = 24, freq='H')
dew_forecast = dew_m.predict(market_future)
dew_forecast = dew_forecast.tail(24)

hum_future = hum_m.make_future_dataframe(periods = 24, freq='H')
hum_forecast = hum_m.predict(market_future)
hum_forecast = hum_forecast.tail(24)

kpa_future = kpa_m.make_future_dataframe(periods = 24, freq='H')
kpa_forecast = kpa_m.predict(market_future)
kpa_forecast = kpa_forecast.tail(24)

#print(market_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(24))
print('Done.')

#combine all forecasts into one dataframe
print('Concatenating Forecasts...')
forecasts = pd.concat([market_forecast['ds'],
						market_forecast['yhat'],
						ontario_forecast['yhat'],
						temp_forecast['yhat'],
						dew_forecast['yhat'],
						hum_forecast['yhat'],
						kpa_forecast['yhat']],
						axis = 1, 
						join_axes = [market_forecast.index])

forecasts.columns = ['ds','Market Demand', 'Ontario Demand', 'Temp (°C)','Dew Point Temp (°C)','Rel Hum (%)','Stn Press (kPa)']
#print(forecasts.info(verbose=True))
print('Done.')

print('Plotting Forecasts...')
print(forecasts['ds'])
forecasts['Hour'] = np.arange(1,25)

forecasts.set_index('Hour', inplace = True, drop = False)
del forecasts['Hour']

fig = plt.figure(figsize=(100,20))
#fig.subplots_adjust(hspace=0.9, wspace=0.9)

grid = plt.GridSpec(2, 4, wspace=0.3, hspace=0.3)

for i in range(1, 5):
    plt.subplot(grid[0,i-1])
    plt.plot(forecasts[forecasts.columns[i]])
    plt.title(forecasts.columns[i])
    plt.xlabel('Hour')

for i in range(5, 7):
    plt.subplot(grid[1,i-5])
    plt.plot(forecasts[forecasts.columns[i]])
    plt.title(forecasts.columns[i])
    plt.xlabel('Hour')


print('Done.')

'''
...............................................................................
[END] FBPROPHET TRAINING / FORECASTING TOMORROWS VALUES
OUTPUT: forecasts ('market', 'ontario', 'temp','dew','hum','kpa')
...............................................................................
'''
'''
...............................................................................
[START] XBoost training and forecast classification
INPUT: data,y, forecasts
...............................................................................

'''

print('Training Classification Model...')

#add time features
forecasts['Hour'] = forecasts['ds'].apply(lambda x: x.hour)
forecasts['Month'] = forecasts['ds'].apply(lambda x: x.month)
forecasts['Day'] = forecasts['ds'].apply(lambda x: x.day)
forecasts['Day of Week'] = forecasts['ds'].apply(lambda x: x.dayofweek)

#del forecasts['ds']

#train model
params = {
    'objective': 'binary:logistic',
    'max_depth': 4,
    'learning_rate': 0.3, 
    'silent': 1.0,
    'scale_pos_weight': float(np.sum(y == 0)) / np.sum(y == 1),
    'n_estimators': 75
}

model = XGBClassifier(**params)

model.fit(data,y)
print('Done.')

print('Classifying Forecast...')

#reorder columns to match data
forecasts = forecasts[['Hour', 'Market Demand', 'Ontario Demand', 'Month', 'Day', 'Temp (°C)', 'Dew Point Temp (°C)', 'Rel Hum (%)', 'Stn Press (kPa)', 'Day of Week']]
pred = model.predict(forecasts)
pred_prob = model.predict_proba(forecasts)

print(pred)
print(pred_prob)

#plot probabilities to grid
plt.subplot(grid[1,2:])
sns.barplot(np.arange(1,25),pred_prob[:,1], color = '#3B5998')
plt.title('Demand Day Probablility by Hour for {}-{}'.format(forecasts['Month'][1],forecasts['Day'][1]))
plt.xlabel('Hour')
plt.ylabel('Probablility')

plt.tight_layout()
plt.show()

np.savetxt("pred.csv", pred, delimiter = ",")
np.savetxt("pred_prob.csv", pred_prob, delimiter = ",")