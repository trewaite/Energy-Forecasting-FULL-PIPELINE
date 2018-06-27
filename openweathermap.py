import numpy as np
import pandas as pd
import openweathermapy.core as own
import datetime
from datetime import datetime as dt

numyears = 3
year_now = int(datetime.datetime.today().strftime('%Y'))

#get historical data 
s = dt(year_now-numyears,1,1).timestamp()
e = dt(year_now,12,31)
data = own.get_history("Toronto,CA", APPID = 'f9eb631697aa43b378cb61590782bd59', units='metric',start=s,end=e)

#change from dt to datetime
conv = {"dt": lambda ts: str(dt.utcfromtimestamp(ts))}

#choose datapoints
selection = data.select(['dt','main.temp','main.humidity','main.pressure'], converters=conv)

df= pd.DataFrame(selection,columns=['Date','Temp','Hum','Press'])

print(df.head())

