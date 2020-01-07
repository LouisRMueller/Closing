from fun_VerificationPlots import *
from cls_ClosingCalc import *
from collections import deque


pd.set_option('display.width', 180)
pd.set_option("display.max_columns", 8)

########################################################################
file_snapshots = os.getcwd() + "\\Data\\20190315_closing_orders.csv"
file_prices = os.getcwd() + "\\Data\\orders_closing_prices.csv"
mode = 'Sensitivity'
percent = np.arange(0, 0.3, 0.05)
########################################################################


Sens = SensitivityAnalysis(file_snapshots)

dump = Sens.sens_analysis(key='bid_limit', percents=percent)
tmp = dump['2019-03-15']['UBSG'][0]
print(tmp.head())

def plot_closing_orders(dump, stock, date='2019-03-15'):
	dic = dump[date][stock]

	for p in [0]:
		df = dic[p][['cumulative bids', 'cumulative asks']].stack()
		print(df.head())

		df = df.reset_index(drop=False)
		df.columns = ['price', 'side','shares']
		df.sort_values('price', ascending=True, inplace=True)
		df['price'] = df['price'].astype(object)
		fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

		sns.lineplot(ax=ax, data=df, x='price', y='shares', hue='side', palette='Set1')
		ax.xaxis.set_major_locator(ticker.MaxNLocator(n=10))

		plt.show()
		plt.close()

plot_closing_orders(dump, 'UBSG')

# dump = Sens.sens_analysis(key='ask_limit', percents=percent, dump=True)
# dump = Sens.sens_analysis(key='all_limit', percents=percent, dump=True)
# Sens.export_results('Verification_v1', 'csv')




print("<<< Sensitivity Sens complete >>>")


