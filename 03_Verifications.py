from cls_ClosingCalc import *
from cls_Visualization_December import *
from fun_VerificationPlots import *

pd.set_option('display.width', 200)
pd.set_option("display.max_columns", 18)

#%%

########################################################################
file_snapshots = os.getcwd() + "\\Data\\20190315_closing_orders.csv"
file_prices = os.getcwd() + "\\Data\\orders_closing_prices.csv"
mode = 'Sensitivity'
percent = np.arange(0, 0.3, 0.05)
########################################################################


Sens = SensitivityAnalysis(file_snapshots, base='CrossedVolume', perc=percent)

test = Sens.process()

raise ValueError

tmp = bid_dump['2019-03-15']['UBSG'][0]

for T in ['ABBN', 'CSGN','NESN','NOVN','ROG','UBSG']:
	plot_closing_orders(bid_dump, T, 'bids only')
	plot_closing_orders(ask_dump, T, 'asks only')
	plot_closing_orders(all_dump, T, 'bids + asks')

Sens.export_results('Verification_v2', 'csv')

#%%

########################################################################
file_bcs = os.getcwd() + "\\Data\\bluechips.csv"
mode, granularity = 'Sensitivity', 'fine'
########################################################################

if mode == 'Sensitivity':
	file_data = os.getcwd() + "\\Exports\\Sensitivity_{}_v2.csv".format(granularity)
	Sens = SensAnalysis(file_data, file_bcs)

	Sens.plt_cont_rmv_indiv_v2('bid_limit')
	Sens.plt_cont_rmv_indiv_v2('ask_limit')
	Sens.plt_cont_rmv_indiv_v2('all_limit')
