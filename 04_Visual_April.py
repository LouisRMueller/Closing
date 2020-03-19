from cls_Visualization_April import *

pd.set_option('display.width', 180)
pd.set_option("display.max_columns", 8)
datadir = os.getcwd() + "\\Data"


########################################################################
file_bcs = os.getcwd() + "\\Data\\bluechips.csv"
mode, granularity = 'Sensitivity', 'rough'
# mode, granularity = 'Sensitivity', 'fine'
# mode, granularity = 'Discovery', None
# mode, granularity = 'Intervals', None
# base = 'SeparateLiquidity' # Limiting to  {'SeparateLiquidity','FullLiquidity','CrossedVolume'}
# base = 'FullLiquidity' # Limiting to  {'SeparateLiquidity','FullLiquidity','CrossedVolume'}
base = 'CrossedVolume' # Limiting to  {'SeparateLiquidity','FullLiquidity','CrossedVolume'}
########################################################################

if mode == 'Sensitivity':
	file_data = os.getcwd() + "\\Exports\\Sensitivity_{}_v3.csv".format(granularity)
	Sens = SensVisual(datapath=file_data, base=base)

	if granularity == 'rough':
		Sens.plot_removed_time(save=True, show=False)

	elif granularity == 'fine':
		Sens.plt_cont_rmv_indiv('bid_limit')
		Sens.plt_cont_rmv_indiv('ask_limit')
		Sens.plt_cont_rmv_indiv('all_limit')
		Sens.plt_cont_rmv_agg()

elif mode == 'Discovery':
	file_data = os.getcwd() + "\\Exports\\Price_Discovery_v3.csv"
	Disc = DiscoVisual(file_data, file_bcs)
	Disc.plot_oib_returns()

elif mode == 'Intervals':
	file_data = os.getcwd() + "\\Exports\\Intervals_v3.csv"
	Inter = IntervalVisual(file_data, file_bcs)
	df = Inter.return_data()

print('--- FINISHED ---')


