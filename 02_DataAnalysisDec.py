from cls_Visualization_December import *

pd.set_option('display.width', 180)
pd.set_option("display.max_columns", 8)
datadir = os.getcwd() + "\\Data"


########################################################################
file_bcs = os.getcwd() + "\\Data\\bluechips.csv"
mode, granularity = 'Sensitivity', 'rough'
# mode, granularity = 'Sensitivity', 'fine'
# mode, granularity = 'Discovery', None
########################################################################

if mode == 'Sensitivity':
	file_data = os.getcwd() + "\\Exports\\Sensitivity_{}_v1.csv".format(granularity)
	Sens = SensAnalysis(file_data, file_bcs)

	if granularity == 'rough':
		Sens.plt_rmv_limit_aggregated()
		Sens.plt_rmv_market_orders()
		Sens.plt_rmv_limit_quant()

	elif granularity == 'fine':
		Sens.plt_cont_rmv_indiv('bid_limit')
		Sens.plt_cont_rmv_indiv('ask_limit')
		Sens.plt_cont_rmv_indiv('all_limit')
		Sens.plt_cont_rmv_agg()



elif mode == 'Discovery':
	file_data = os.getcwd() + "\\Exports\\Price_Discovery_v1.csv"
	Disc = DiscoAnalysis(file_data, file_bcs)
	Disc.plt_disco_distr_xsect()
	Disc.plt_closing_volume()
	Disc.plt_deviation_discovery()
	Disc.plt_disco_by_title(5)


print('--- FINISHED ---')
	
	
