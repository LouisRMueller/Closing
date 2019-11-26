import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

pd.set_option('display.width', 180)
pd.set_option("display.max_columns", 8)
datadir = os.getcwd() + "\\Data"

bcs = pd.read_csv(datadir + "\\bluechips.csv", index_col='symbol')
bcs.sort_index(inplace=True)



PS = pd.read_csv(os.getcwd() + "\\Exports\\Price_Discovery_v1.csv", index_col=['Date', 'Symbol'])
PS.xs()

SE = pd.read_csv(os.getcwd() + "\\Exports\\Sensitivity_fine_v1.csv", index_col=['Mode', 'Date', 'Symbol', 'Percent'])
SE = pd.read_csv(os.getcwd() + "\\Exports\\Sensitivity_rough_v1.csv", index_col=['Mode', 'Date', 'Symbol', 'Percent'])

misprice = PS['actual_close_price'] - PS['pre_midquote']

misprice.groupby('Date').mean().plot()
plt.show()