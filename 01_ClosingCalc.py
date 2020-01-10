from cls_ClosingCalc import *

pd.set_option('display.width', 180)
pd.set_option("display.max_columns", 8)

########################################################################
file_snapshots = os.getcwd() + "\\Data\\orders_close_closing_main_v2.csv"
file_prices = os.getcwd() + "\\Data\\orders_closing_prices.csv"
mode = 'Sensitivity'
granularity = 'fine'
########################################################################


if mode == 'Sensitivity':
	Sens = SensitivityAnalysis(file_snapshots)
	if granularity == 'rough':
		percent = np.arange(0, 0.55, 0.05)
	elif granularity == 'fine':
		percent = np.arange(0, 0.21, 0.01)
	else:
		raise ValueError("Wrong input for granularity.")
		
	Sens.sens_analysis(key='bid_limit', percents=percent)
	Sens.sens_analysis(key='ask_limit', percents=percent)
	Sens.sens_analysis(key='all_limit', percents=percent)
	Sens.sens_analysis(key='all_market')
	Sens.export_results('Sensitivity_{}_v2'.format(granularity), 'csv')
	print("<<< Sensitivity Sens complete >>>")

 
elif mode == 'Discovery':
	Discovery = PriceDiscovery(file_snapshots, file_prices)
	Discovery.discovery_analysis()
	Discovery.export_results('Price_Discovery_v2', 'csv')
	print("<<< Price Disc complete >>>")


