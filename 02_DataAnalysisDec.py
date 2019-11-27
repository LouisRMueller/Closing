import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from Classes_DataAnalysis import *

pd.set_option('display.width', 180)
pd.set_option("display.max_columns", 8)
datadir = os.getcwd() + "\\Data"


########################################################################
file_bcs = os.getcwd() + "\\Data\\bluechips.csv"
mode = 'Test'
granularity = 'fine'
########################################################################

if mode == 'Sensitivity':
	file_data = os.getcwd() + "\\Exports\\Sensitivity_{}_v1.csv".format(granularity)
	Analysis = SensAnalysis(file_data, file_bcs)
	df = Analysis._raw_data

	if granularity == 'rough':
		pass

elif mode == 'Discovery':
	file_data = os.getcwd() + "\\Exports\\Price_Discovery_v1.csv"
	Discovery = DiscoAnalysis(file_data, file_bcs)


	
	
