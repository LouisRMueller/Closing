import pandas as pd
import numpy as np
import os

PS = pd.read_csv(os.getcwd() + "\\Export\\Price_Discovery_v1.csv", index_col=['Date', 'Symbol'])

SE = pd.read_csv(os.getcwd() + "\\Export\\Sensitivity_fine_v1.csv", index_col=['Mode', 'Date', 'Symbol', 'Percent'])
SE = pd.read_csv(os.getcwd() + "\\Export\\Sensitivity_rough_v1.csv", index_col=['Mode', 'Date', 'Symbol', 'Percent'])

## Rebased version