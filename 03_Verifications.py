from Concurrent_ClosingCalcs import *
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

priceData = Sens.process()['all_limit'][0]
closingInfo = Sens._result_dict[('ask_limit', '2019-03-15', 'NESN', 0.0)]

Sens.plotClosingUncross()


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
