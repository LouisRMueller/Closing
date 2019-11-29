from Classes_DataAnalysis import *

pd.set_option('display.width', 180)
pd.set_option("display.max_columns", 8)
datadir = os.getcwd() + "\\Data"


########################################################################
file_bcs = os.getcwd() + "\\Data\\bluechips.csv"
mode = 'Test'
granularity = 'rough'
########################################################################

if mode == 'Sensitivity':
	file_data = os.getcwd() + "\\Exports\\Sensitivity_{}_v1.csv".format(granularity)
	Sens = SensAnalysis(file_data, file_bcs)

	if granularity == 'rough':
		pass
	elif granularity == 'fine':
		Sens.plt_cont_rmv_indiv('bid_limit')
		Sens.plt_cont_rmv_indiv('ask_limit')
		Sens.plt_cont_rmv_indiv('all_limit')

elif mode == 'Discovery':
	file_data = os.getcwd() + "\\Exports\\Price_Discovery_v1.csv"
	Discovery = DiscoAnalysis(file_data, file_bcs)


	
	
