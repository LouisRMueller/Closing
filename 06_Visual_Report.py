from cls_Visualization_April import *

pd.set_option('display.width', 180)
pd.set_option("display.max_columns", 8)
datadir = os.getcwd() + "\\Data"

########################################################################
file_returns = os.getcwd() + "\\Data\\end_of_day_returns_bluechips.csv"
mode = 'Discovery'
bases = {'SeparateOrders', 'FullLiquidity', 'CrossedVolume'}
########################################################################

print(mode)

if mode == 'Sensitivity':
	for base in bases:
		print(base)
		file_data = os.getcwd() + "\\Exports\\Sensitivity_rough_{b}_v3.csv".format(b=base)
		
		Sens = SensVisual(datapath=file_data, base=base)
		Sens.plots_report(save=True, show=True)

elif mode == 'Discovery':
	file_data = os.getcwd() + "\\Exports\\Price_Discovery_v3.csv"
	Disc = DiscoVisual(file_data, file_returns)
	Disc.plots_report(save=True, show=False)

elif mode == 'Intervals':
	file_data = os.getcwd() + "\\Exports\\Intervals_v3.csv"
	Inter = IntervalVisual(file_data)
	Inter.plots_report(save=True, show=False)
	# Inter.plot_months_lags(save=True, show=False)

print('--- FINISHED ---')
