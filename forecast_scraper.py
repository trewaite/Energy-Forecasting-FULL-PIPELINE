import pandas as pd
import numpy as np

import sys, os, subprocess
import wget
import datetime
import time
from xml.dom import minidom


def pull_ieso_forecast():

	#get todays forecast report (posted by 9:30am)
	day = int(datetime.datetime.today().strftime('%d'))
	month = datetime.datetime.today().strftime('%m')
	wget.download("http://reports.ieso.ca/public/OntarioZonalDemand/PUB_OntarioZonalDemand_2018{}{}.xml".format(month,day),out='wget_data/Forecast')
	dfcols = ['Date','Hour','Ontario']
	df_xml = pd.DataFrame(columns=dfcols, dtype = float)

	xmldoc = minidom.parse('wget_data/Forecast/PUB_OntarioZonalDemand_2018{}{}.xml'.format(month,day))

	document = xmldoc.getElementsByTagName('Document')[0]
	docbody = document.getElementsByTagName('DocBody')[0]

	for i in range(0,35):

		zonaldemands = docbody.getElementsByTagName('ZonalDemands')[i]#enemerate this for all days
		zonaldemand = zonaldemands.getElementsByTagName('ZonalDemand')[0]
		ontario = zonaldemand.getElementsByTagName('Ontario')[0]
		demand = ontario.getElementsByTagName('Demand')

		date = zonaldemands.getElementsByTagName('DeliveryDate')[0].firstChild.data

		for demand in demand:
			hour = demand.getElementsByTagName('DeliveryHour')[0].firstChild.data
			energy = demand.getElementsByTagName('EnergyMW')[0].firstChild.data
			df_xml = df_xml.append(pd.Series([date, hour, energy], index=dfcols), ignore_index=True)


	df_xml.to_csv('wget_data/Forecast/PUB_OntarioZonalDemand_2018{}{}.csv'.format(month,day), index = False)

	df_xml['Date'] = pd.to_datetime(df_xml['Date']) + pd.to_timedelta(df_xml['Hour'].apply(lambda x: int(x)), unit = 'h')
	
	df_xml.set_index('Date',inplace = True, drop = False)
	df_xml['Date'] = df_xml['Date'].dt.date
	
	df_xml.reset_index(drop=True, inplace=True)

	df_xml['Date'] = pd.to_datetime(df_xml['Date'])
	df_xml['Hour'] = df_xml['Hour'].apply(lambda x: int(x))
	df_xml['Ontario'] = df_xml['Ontario'].apply(lambda x: int(x))

	return(df_xml)

#print(pull_ieso_forecast())