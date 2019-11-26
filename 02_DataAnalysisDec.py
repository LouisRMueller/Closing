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
mode = 'Sensitivity'
granularity = 'fine'
########################################################################

if mode == 'Sensitivity':
	file_data = os.getcwd() + "\\Exports\\Sensitivity_{}_v1.csv".format(granularity)
	Analysis = SensAnalysis(file_data, file_bcs)
	print(Analysis._data.head(50))


	if granularity == 'rough':
		pass
	
	
	
