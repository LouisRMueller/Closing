import pandas as pd
from pandas.plotting import register_matplotlib_converters
import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib import dates
import seaborn as sns
import copy
import gc


# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')

register_matplotlib_converters()

conv = 2.54
dpi = 400
figsize = (22 / conv, 13 / conv)
figdir = os.getcwd() + "\\01 Presentation December\\Figures"


class DataAnalysis:
	def __init__(self, datapath, bluechippath):
		self._bluechips = pd.read_csv(bluechippath)['symbol']
		self._raw_data = pd.read_csv(datapath, parse_dates=['Date'])
		self._figpath = os.getcwd() + "\\01 Presentation December"

	def _raw_to_bcs(self):
		self._bcs_data = self._raw_data[self._raw_data.index.get_level_values(level='Symbol').isin(self._bluechips)]


class SensAnalysis(DataAnalysis):
	def __init__(self, datapath, bluechippath):
		super().__init__(datapath, bluechippath)
		self._raw_data['Percent'] = self._raw_data['Percent'].round(2)
		self._raw_data.set_index(['Mode', 'Date', 'Symbol', 'Percent'], inplace=True)
		self._raw_to_bcs()

	def plot_cont_sens(self):
		self._raw_data

	def plot_remove_limit_individual(self, stock, mode):
		limit = 0.35
		namedict = dict(bid_limit="bid limit orders", ask_limit="ask limit orders", all_limit="all limit orders")
		locdict = dict(bid_limit='lower center', ask_limit='upper center', all_limit='upper center')

		tmp = self._bcs_data.loc[mode, :].xs(stock, level='Symbol')
		tmp = (tmp['adj_price'] - tmp['close_price']) / tmp['close_price'] * 10000
		tmp = tmp.unstack(level='Percent', fill_value=np.nan)
		tmp = tmp.iloc[:, tmp.columns <= limit]

		tmp.plot(figsize=figsize, linewidth=1)
		plt.hlines(0, 0, len(tmp.index), 'k', 'dashed', linewidth=1)
		plt.xlabel("")
		plt.ylabel("Deviation from closing price in bps")
		plt.title("{}: Gradual removal of {}".format(stock, namedict[mode]))

		plt.legend(loc=locdict[mode], ncol=int(len(tmp.columns)),
				 labels=[str(int(x * 100)) + " \%" for x in tmp.columns])

		plt.tight_layout()
		plt.savefig(figdir + "\\SensitivityRough\\remove_{}_{}".format(mode, stock), dpi=dpi)
		plt.close()

	def plt_rmv_limit_aggregated(self, mode, aggreg):
		limit = 0.35
		namedict = dict(bid_limit="bid limit orders", ask_limit="ask limit orders", all_limit="all limit orders")
		aggdict = dict(mean='Average', median='Median')
		locdict = dict(bid_limit='lower center', ask_limit='upper center', all_limit='upper center')
		tmp = copy.deepcopy(self._bcs_data.loc[mode, :])

		if aggreg == 'mean':
			tmp = tmp.groupby(['Date', 'Percent']).mean()
		elif aggreg == 'median':
			tmp = self._bcs_data.loc[mode, :].groupby(['Date', 'Percent']).median()

		tmp = (tmp['adj_price'] - tmp['close_price']) / tmp['close_price'] * 10000
		tmp = tmp.unstack(level='Percent', fill_value=np.nan)
		tmp = tmp.loc[:, tmp.columns <= limit]

		tmp.plot(figsize=figsize, linewidth=1)
		plt.hlines(0, 0, len(tmp.index), 'k', 'dashed', linewidth=1)
		plt.xlabel("")
		plt.ylabel("Deviation from closing price in bps")
		plt.title("{} impact of gradual removal of {} across SLI".format(aggdict[aggreg], namedict[mode]))

		plt.legend(loc=locdict[mode], ncol=int(len(tmp.columns)),
				 labels=[str(int(x * 100)) + " \%" for x in tmp.columns])

		plt.tight_layout()
		plt.savefig(figdir + "\\SensitivityRough\\agg_remove_{}_{}".format(mode, aggreg), dpi=dpi)
		plt.show()
		print(tmp.head())

		return tmp

	def plt_rmv_limit_quant(self, mode):
		limit = 0.3
		namedict = dict(bid_limit="bid limit orders", ask_limit="ask limit orders", all_limit="all limit orders")
		aggdict = dict(mean='Average', median='Median')
		locdict = dict(bid_limit='lower center', ask_limit='upper center', all_limit='upper center')
		raw = copy.deepcopy(self._bcs_data.loc[mode, :])

		quants = raw.groupby(['Date', 'Symbol']).first()
		quants = quants.groupby('Date')['close_price'].transform(lambda x: pd.qcut(x, 3, labels=range(1,4)))
		quants.rename('quantile', inplace=True)

		# tmp = pd.concat([raw, quants], axis=0, join='outer')
		tmp = raw[['close_price', 'adj_price']].join(quants, on=['Date','Symbol'], how='left', )


		tmp = tmp.groupby(['quantile', 'Date', 'Percent']).mean()

		tmp = (tmp['adj_price'] - tmp['close_price']) / tmp['close_price'] * 10000
		tmp = tmp.unstack(level='Percent', fill_value=np.nan)
		tmp = tmp.loc[:, tmp.columns <= limit]



		fig, axes = plt.subplots(3,1, figsize=figsize, sharex=True)


		for ax,qt in zip(axes, range(1,4)):
			plot_df = tmp.xs(qt, level='quantile')
			ax.plot(plot_df)
			ax.hlines(0, 0, plot_df.shape[0])


		# locator = dates.AutoDateLocator()
		# axes[-1].xaxis.set_major_locator(locator)
		# formatter = dates.ConciseDateFormatter(locator)
		# axes[-1].xaxis.set_major_formatter(formatter)

		plt.show()
		plt.close(fig)
		fig.clf()
		gc.collect()

		return tmp.xs(qt, level='quantile')



class DiscoAnalysis(DataAnalysis):
	def __init__(self, datapath, bluechippath):
		super().__init__(datapath, bluechippath)
		self._raw_data.set_index(['Date', 'Symbol'], inplace=True)
		self._raw_to_bcs()

	def plot_closings_per_stock(self):
		limit = 50

		meanvol = self._raw_data.loc[:, 'close_vol'].groupby('Symbol').mean()
		tmp_vol = meanvol / meanvol.max()
		tmp_vol.sort_values(ascending=False, inplace=True)
		tmp_vol = tmp_vol[tmp_vol > 0]

		# # Density Function of Closing frequencies
		# tmp_df = self._raw_data.loc[:, 'close_price_calculated'].groupby('Symbol').count()
		# numdays = self._raw_data.index.get_level_values(level='Date').nunique()
		# tmp_norm = tmp_df / numdays * 100
		# tmp_norm = tmp_norm[tmp_vol.index]
		# fig, ax = plt.subplots(1, 1, figsize=figsize)
		# ax.hist(tmp_norm.values, bins=100, density=True)
		# plt.show()

		# Volume contribution among all stocks without labels
		fig, ax = plt.subplots(1, 1, figsize=figsize)
		ax.bar(np.arange(1, len(tmp_vol) + 1), tmp_vol.values, width=1)
		ax.set_yscale('log')
		ax.set_ylabel("Volume divided by largest")
		ax.set_xlabel("Sorted titles")
		ax.set_title("Average daily closing volume normalized")
		plt.savefig(figdir + "\\PriceDiscovery\\volume_percent_aggregated", dpi=dpi)
		plt.close()

		# Volume percentage of closings with labels
		fig, ax = plt.subplots(1, 1, figsize=figsize)
		ax.bar(tmp_vol.index[:limit], tmp_vol[:limit].values, width=0.7)
		ax.tick_params(axis='x', rotation=90)
		ax.set_title("Average daily closing volume normalized for {} most liquid titles".format(str(limit)))
		ax.set_ylabel("Volume divided by largest")
		plt.savefig(figdir + "\\PriceDiscovery\\volume_percent_stocks", dpi=dpi)
		plt.close()

		print(tmp_vol.tail(10))


file_data = os.getcwd() + "\\Exports\\Sensitivity_rough_v1.csv"
file_bcs = os.getcwd() + "\\Data\\bluechips.csv"
Sens = SensAnalysis(file_data, file_bcs)
tp = Sens.plt_rmv_limit_quant('all_limit')
