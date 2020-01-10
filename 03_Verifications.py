from fun_VerificationPlots import *
from cls_ClosingCalc import *
# from cls_DataAnalysis import *

pd.set_option('display.width', 180)
pd.set_option("display.max_columns", 8)

########################################################################
file_snapshots = os.getcwd() + "\\Data\\20190315_closing_orders.csv"
file_prices = os.getcwd() + "\\Data\\orders_closing_prices.csv"
mode = 'Sensitivity'
percent = np.arange(0, 0.3, 0.05)
########################################################################


Sens = SensitivityAnalysis(file_snapshots)

bid_dump = Sens.sens_analysis(key='bid_limit', percents=percent)
ask_dump = Sens.sens_analysis(key='ask_limit', percents=percent)
all_dump = Sens.sens_analysis(key='all_limit', percents=percent)

tmp = bid_dump['2019-03-15']['UBSG'][0]

#%%

for T in ['ABBN', 'CSGN','NESN','NOVN','ROG','UBSG']:
	plot_closing_orders(bid_dump, T, 'bids only')
	plot_closing_orders(ask_dump, T, 'asks only')
	plot_closing_orders(all_dump, T, 'bids + asks')

Sens.export_results('Verification_v2', 'csv')


raise ValueError

########################################################################
file_bcs = os.getcwd() + "\\Data\\bluechips.csv"
mode, granularity = 'Sensitivity', 'rough'
# mode, granularity = 'Sensitivity', 'fine'
# mode, granularity = 'Discovery', None
########################################################################

if mode == 'Sensitivity':
	file_data = os.getcwd() + "\\Exports\\Sensitivity_{}_v1.csv".format(granularity)
	Sens = SensAnalysis(file_data, file_bcs)
