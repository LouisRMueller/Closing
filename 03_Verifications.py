from fun_VerificationPlots import *
from cls_ClosingCalc import *


pd.set_option('display.width', 180)
pd.set_option("display.max_columns", 8)

########################################################################
file_snapshots = os.getcwd() + "\\Data\\20190315_closing_orders.csv"
file_prices = os.getcwd() + "\\Data\\orders_closing_prices.csv"
mode = 'Sensitivity'
percent = np.arange(0, 0.3, 0.05)
########################################################################


Sens = SensitivityAnalysis(file_snapshots)

dump = Sens.sens_analysis(key='bid_limit', percents=percent, dump=True)


def plot_closing_orders(dump, stock, date='2019-03-15'):
	dic = dump[date][stock]

	for p in iter(dic):
		df = dic[p].stack()
		df = df.reset_index(drop=False)
		df.columns = ['price', 'side','shares']
		df['price'] = df['price'].astype(object)
		print(df.dtypes)
		fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

		sns.catplot(ax=1, data=df, x='price', y='shares', kind='bar')
		plt.show()
		plt.close()

plot_closing_orders(dump, 'UBSG')

# dump = Sens.sens_analysis(key='ask_limit', percents=percent, dump=True)
# dump = Sens.sens_analysis(key='all_limit', percents=percent, dump=True)
# Sens.export_results('Verification_v1', 'csv')




print("<<< Sensitivity Sens complete >>>")


