from cls_ClosingCalc import *

pd.set_option('display.width', 180)
pd.set_option("display.max_columns", 8)

########################################################################
file_snapshots = os.getcwd() + "\\Data\\orders_close_closing_main_v3.csv"
file_prices = os.getcwd() + "\\Data\\orders_closing_prices.csv"
base = 'TotalVolume'  # ('LimitOrders', ''TotalVolume')
mode = 'Sensitivity'
granularity = 'fine'
########################################################################


if mode == 'Sensitivity':
	Sens = SensitivityAnalysis(file_snapshots)
	if granularity == 'rough':
		percent = np.arange(0, 0.55, 0.05).round(2)
	elif granularity == 'fine':
		percent = np.arange(0, 0.41, 0.01).round(2)
	else:
		raise ValueError("Wrong input for granularity.")

	Sens.sensitivity_processing(key='bid_limit', percents=percent, remove_func=base)
	Sens.sensitivity_processing(key='ask_limit', percents=percent, remove_func=base)
	Sens.sensitivity_processing(key='all_limit', percents=percent, remove_func=base)
	Sens.sensitivity_processing(key='all_market', remove_func=base) if base == 'LimitOrders' else None
	Sens.sensitivity_processing(key='cont_market', remove_func=base) if base == 'LimitOrders' else None

	Sens.export_results('Sensitivity_{}_{}_v3'.format(granularity, base), 'csv')

	print("<<< Sensitivity Sens complete >>>")


elif mode == 'Discovery':
	Discovery = PriceDiscovery(file_snapshots, file_prices)
	Discovery.discovery_processing()
	Discovery.export_results('Price_Discovery_v3', 'csv')
	print("<<< Price Discovery complete >>>")


elif mode == 'Intervals':
	Intervals = IntervalAnalysis(file_snapshots)
	Intervals.interval_processing()

	Intervals.export_results('Intervals_v3', 'csv')
	print("<<< Intervals complete >>>")
