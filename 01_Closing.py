import os
import pandas as pd
import numpy as np
from Class_SnapBook import SnapBook

pd.set_option('display.width', 180)
pd.set_option("display.max_columns", 8)

file = os.getcwd() + "\\Data\\orders_close_closing_main.csv"
SB = SnapBook(file)


percent = [0.05, 0.10, 0.15, 0.20, 0.25]
SB.process_analysis(key='bid_limit', percents=percent)
print("finish")

#%%

df = SB.get_output()

pd.DataFrame.from_dict(df, orient='index')

pd.Panel(df)