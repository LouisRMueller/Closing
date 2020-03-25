from cls_ClosingCalc import *

pd.set_option('display.width', 180)
pd.set_option("display.max_columns", 8)

########################################################################
file_snapshots = os.getcwd() + "\\Data\\orders_close_closing_main_v3.csv"
file_prices = os.getcwd() + "\\Data\\closing_prices.csv"
base = 'SeparateOrders'  # Limiting to  {'SeparateOrders','FullLiquidity','CrossedVolume'}
mode = 'Sensitivity'
granularity = 'rough'
########################################################################


if mode == 'Sensitivity':
	print("Base: {}".format(base))
	if granularity == 'rough':
		percent = np.arange(0, 0.88, 0.05).round(2)
	elif granularity == 'fine':
		percent = np.arange(0, 0.41, 0.01).round(2)
	else:
		raise ValueError("Wrong input for granularity.")

	Sens = SensitivityAnalysis(file_snapshots, base=base, perc=percent)
	Sens.process()
	Sens.export_results('Sensitivity_{}_{}_v3'.format(granularity, base), 'csv')

	print("<<< {1} Sensitivity {0} complete >>>".format(granularity, base))


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
