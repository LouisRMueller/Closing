import os
import pandas as pd
import numpy as np
from Class_Sensitivity import SensitivityAnalysis
from Class_PriceDiscovery import PriceDiscovery

pd.set_option('display.width', 180)
pd.set_option("display.max_columns", 8)

file_snapshots = os.getcwd() + "\\Data\\orders_close_closing_main.csv"
file_prices = os.getcwd() + "\\Data\\orders_closing_prices.csv"


Discovery = PriceDiscovery(file_snapshots, file_prices)
Discovery.discovery_analysis()
df = Discovery.results_to_df()
Discovery.export_results('Price_Discovery_v1', 'csv')
Discovery.export_results('Price_Discovery_v1', 'xlsx')


#%%
#
# Sens = SensitivityAnalysis(file_snapshots)
#
# percent = np.arange(0.05, 0.55, 0.05)
# Sens.sens_analysis(key='bid_limit', percents=percent)
# Sens.sens_analysis(key='ask_limit', percents=percent)
# Sens.sens_analysis(key='all_limit', percents=percent)
# Sens.sens_analysis(key='all_market')
# print("finished")

#%%

