from cls_Visualization_April import *

pd.set_option('display.width', 180)
pd.set_option("display.max_columns", 8)
datadir = os.getcwd() + "\\Data"

########################################################################
file_returns = os.getcwd() + "\\Data\\end_of_day_returns_bluechips.csv"
mode = 'Intervals'
bases = {'SeparateOrders', 'FullLiquidity', 'CrossedVolume'}
########################################################################
print(mode)

if mode == 'Sensitivity':
	for base in {'CrossedVolume'}:
		print(base)
		file_data = os.getcwd() + "\\Exports\\Sensitivity_rough_{b}_v3.csv".format(b=base)
		
		Sens = SensVisual(datapath=file_data, base=base)
		Sens.plot_removal_time(save=True, show=False)
		Sens.plot_removal_quantiles(save=True, show=False)
		Sens.plot_removal_by_stock(stock='NESN', show=True)

elif mode == 'Discovery':
	file_data = os.getcwd() + "\\Exports\\Price_Discovery_v3.csv"
	Disc = DiscoVisual(file_data, file_returns)
	Disc.plot_oib_returns(save=True, show=False)
	Disc.plots_wpdc(save=True, show=False)
	Disc.plot_stocks_time_compare(save=True, show=False)

elif mode == 'Intervals':
	file_data = os.getcwd() + "\\Exports\\Intervals_v3.csv"
	Inter = IntervalVisual(file_data)
	Inter.plot_months_lags(save=True, show=False)
	Inter.plot_stocks_lags(save=True, show=False)
	Inter.plot_stocks_within(nstocks=9, save=True, show=False)

elif mode == 'Extras':
	pricepath = os.getcwd() + "\\03 Presentation April\\Extras\\20200505_bc_closingprices.csv"
	volpath = os.getcwd() + "\\03 Presentation April\\Extras\\20200506_vol_overhang.csv"
	Extras = ExtraVisual(pricepath, volpath)
	df = Extras._prices
	Extras.plot_extras(show=True, save=True)

print('--- FINISHED ---')
