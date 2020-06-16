import os
import pandas as pd

pd.set_option('display.width', 200)
pd.set_option("display.max_columns", 16)

# %%

for base in {'CrossedVolume', 'FullLiquidity', 'SeparateOrders'}:
	raw = pd.read_csv(os.getcwd() + "\\Exports\\Adam_Sens_fine_{}.csv".format(base), index_col=['Mode', 'Date', 'Symbol', 'Percent'])
	df = raw.loc[:, ['close_price', 'adj_price', 'close_vol', 'adj_vol']]
	df['Dev_Price'] = df['adj_price'] - df['close_price']
	df['Dev_Turnover'] = df['adj_vol'] - df['close_vol']
	
	df.loc[['all_limit'], 'Dev_Price'] = df.loc[['all_limit'], 'Dev_Price'].abs()
	
	exp = df.groupby(['Mode', 'Date', 'Percent']).mean()
	exp.to_csv(os.getcwd() + "\\05 Delivery June\\Adam_Sens_fine_{}_aggregated.csv".format(base))

# %%
raw = pd.read_csv(os.getcwd() + "\\05 Delivery June\\Intervals.csv")
raw.groupby(['Lag', 'Symbol']).mean().to_csv(os.getcwd() + "\\05 Delivery June\\Invervals_stocks_Average.csv")
raw.groupby(['Lag', 'Symbol']).median().to_csv(os.getcwd() + "\\05 Delivery June\\Invervals_stocks_Median.csv")
raw.loc[raw['Symbol'] == 'NESN', ['Lag', 'Date' , 'Symbol', 'Month', 'Deviation']].to_csv(os.getcwd() + "\\05 Delivery June\\NESN_Intervals.csv")
