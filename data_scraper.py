import pandas as pd
import numpy as np

import sys, os, subprocess
import wget
import datetime
import time

#Toronto
station_ID = 31688
num_years = 3 


def PullData():

    #Get Datetimes
    year_now = int(datetime.datetime.today().strftime('%Y'))

    #Check directory
    print('Checking for missing data...')

    #clean demand direc
    root_demand = 'wget_data/Demand'
    filenames_demand = []

    for path, subdirs, files in os.walk(root_demand):
        for name in files:
            filenames_demand.append(os.path.join(path, name))

    for i in range(len(filenames_demand)):
        os.remove(filenames_demand[i])
            
    #clean weather direc
    root_weather = 'wget_data/Weather'
    filenames_weather = []

    for path, subdirs, files in os.walk(root_weather):
        for name in files:
            filenames_weather.append(os.path.join(path, name))

    for i in range(len(filenames_weather)):
        os.remove(filenames_weather[i])

    print('Done.')

    #Compare with required
    year_static = list(range(year_now-num_years+1,year_now+1))
  
    #Pull IESO demand data
    print('Pulling IESO Demand Data...')
    for year in year_static:
        print(year)
        wget.download("http://reports.ieso.ca/public/Demand/PUB_Demand_{}.csv".format(year),out='wget_data/Demand')
    print('Done.')

    #Pull Statscan weather data
    print('Pulling Statscan Weather Data for Station {}...'.format(station_ID))

    for year in year_static:
        print(year)
        for month in range(1,12):
            wget.download("http://climate.weather.gc.ca/climate_data/bulk_data_e.html?format=csv&stationID={0}&Year={1}&Month={2}&Day=14&timeframe=1&submit=Download+Data".format(station_ID,year,month),out='wget_data/Weather')

    print('Done.')


    #Clean, Convert CSV to Dataframes

    root = 'wget_data/Demand/'
    filenames = []

    for path, subdirs, files in os.walk(root):
        for name in files:
            filenames.append(os.path.join(path, name))


    print('Reading demand csvs to dataframe...')
    demand_data = pd.concat( [ pd.read_csv(f,skiprows=3) for f in filenames ] )
    print('Done.')

    root = 'wget_data/Weather/'
    filenames = []

    for path, subdirs, files in os.walk(root):
        for name in files:
            filenames.append(os.path.join(path, name))


    print('Reading weather csvs to dataframe...')
    weather_data = pd.concat( [ pd.read_csv(f,skiprows=15) for f in filenames ] )
    print('Done.')

    return demand_data, weather_data
