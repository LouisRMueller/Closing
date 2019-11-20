import os
import pandas as pd
import numpy as np
from Class_Sensitivity import SensitivityAnalysis
from Class_PriceDiscovery import PriceDiscovery

pd.set_option('display.width', 180)
pd.set_option("display.max_columns", 8)



########################################################################
file_snapshots = os.getcwd() + "\\Data\\orders_close_closing_main.csv"
file_prices = os.getcwd() + "\\Data\\orders_closing_prices.csv"
mode = 'Sensitivity'
granularity = 'rough'
########################################################################


if mode == 'Sensitivity':
	Sens = SensitivityAnalysis(file_snapshots)
	if granularity == 'rough':
		percent = np.arange(0.05, 0.55, 0.05)
	elif granularity == 'fine':
		percent = np.arange(0.01, 0.51, 0.01)
	Sens.sens_analysis(key='bid_limit', percents=percent)
	Sens.sens_analysis(key='ask_limit', percents=percent)
	Sens.sens_analysis(key='all_limit', percents=percent)
	Sens.sens_analysis(key='all_market')
	Sens.export_results('Sensitivity_{}_v1'.format(granularity), 'csv', ['Mode', 'Date', 'Symbol', 'Percent'])
	print("<<< Sensitivity Analysis complete >>>")

 
elif mode == 'Discovery':
	Discovery = PriceDiscovery(file_snapshots, file_prices)
	Discovery.get_SB()
	Discovery.discovery_analysis()
	df = Discovery.results_to_df()
	Discovery.export_results('Price_Discovery_v1', 'csv', ['Date', 'Symbol'])
	Discovery.export_results('Price_Discovery_v1', 'xlsx', ['Date', 'Symbol'])
	print("<<< Price Discovery complete >>>")


#%%


#%%

